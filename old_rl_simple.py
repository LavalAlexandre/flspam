"""Simplified GRPO trainer for Phase 2: RL with detector-only reward.

This trainer is used after SFT pre-training on spam data.
It uses only the detector score as reward, with minimal/no prompting.
"""

import heapq
import shutil
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import PeftModel
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .bypass_logger import BypassLogger, clean_response, has_spam_payload, TASK_TAG
from .config import GRPOSpamConfig


class GRPOProfilingCallback(TrainerCallback):
    """Callback to profile GRPO training steps.

    Tracks:
    - Time per training step
    - Generation time (from step start to reward call)
    - Training time (gradient computation, optimization)
    """

    def __init__(self):
        self.step_times = []
        self.step_start = None
        self.total_steps = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self.step_start is not None:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - self.step_start
            self.step_times.append(elapsed)
            self.total_steps += 1

            # Log to wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "profiling/step_time_callback": elapsed,
                    },
                    commit=False,
                )

    def print_summary(self):
        """Print final profiling summary."""
        if self.step_times:
            avg_time = sum(self.step_times) / len(self.step_times)
            min_time = min(self.step_times)
            max_time = max(self.step_times)

            print(f"\n{'=' * 60}")
            print(f"📊 GRPO PROFILING SUMMARY ({self.total_steps} steps)")
            print(f"{'=' * 60}")
            print(f"  Average step time: {avg_time:.2f}s")
            print(f"  Min step time:     {min_time:.2f}s")
            print(f"  Max step time:     {max_time:.2f}s")
            print(f"  Total time:        {sum(self.step_times) / 60:.1f} min")
            print(f"{'=' * 60}\n")


class BypassTrackingCallback(TrainerCallback):
    """Callback to track bypass metrics and save top-k performing models.

    Tracks:
    - Actual bypass count per step (undetected + high judge score)
    - Bypass score = bypass_rate * avg_judge_score
    - Saves top-k models based on bypass score
    """

    def __init__(
        self, reward_func, output_dir: str, top_k: int = 2, judge_threshold: float = 0.5
    ):
        """Initialize bypass tracking.

        Args:
            reward_func: CombinedReward instance to get step metrics
            output_dir: Base directory for saving checkpoints
            top_k: Number of top models to keep
            judge_threshold: Minimum judge score (0-1) to count as quality bypass
        """
        self.reward_func = reward_func
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.judge_threshold = judge_threshold

        # Track step metrics
        self.step_metrics = []

        # Min-heap of (score, step, checkpoint_path) - keeps top_k best
        # Using negative score because heapq is a min-heap
        self.top_models = []

        # Create checkpoints directory
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def compute_bypass_score(self, bypass_rate: float, avg_judge_score: float) -> float:
        """Compute combined bypass score.

        Score = bypass_rate * avg_judge_score
        - bypass_rate: fraction of samples that bypassed detector (0-1)
        - avg_judge_score: average judge score for bypassed samples (0-1)

        High score means: many bypasses AND they are high quality spam.
        """
        return bypass_rate * avg_judge_score

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        # Get metrics from last step stored in reward_func
        step = state.global_step

        # These will be set by the reward function during __call__
        if hasattr(self.reward_func, "last_step_metrics"):
            metrics = self.reward_func.last_step_metrics

            bypass_count = metrics.get("bypass_count", 0)
            quality_bypass_count = metrics.get("quality_bypass_count", 0)
            total_samples = metrics.get("total_samples", 1)
            avg_judge_score = metrics.get("avg_judge_score", 0.5)
            avg_quality_judge_score = metrics.get("avg_quality_judge_score", 0.5)

            bypass_rate = bypass_count / total_samples
            quality_bypass_rate = quality_bypass_count / total_samples
            bypass_score = self.compute_bypass_score(
                quality_bypass_rate, avg_quality_judge_score
            )

            # Store metrics
            step_data = {
                "step": step,
                "bypass_count": bypass_count,
                "quality_bypass_count": quality_bypass_count,
                "total_samples": total_samples,
                "bypass_rate": bypass_rate,
                "quality_bypass_rate": quality_bypass_rate,
                "avg_judge_score": avg_judge_score,
                "avg_quality_judge_score": avg_quality_judge_score,
                "bypass_score": bypass_score,
            }
            self.step_metrics.append(step_data)

            # Log to wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "bypass/count": bypass_count,
                        "bypass/quality_count": quality_bypass_count,
                        "bypass/rate": bypass_rate,
                        "bypass/quality_rate": quality_bypass_rate,
                        "bypass/avg_judge_score": avg_judge_score,
                        "bypass/score": bypass_score,
                    },
                    commit=False,
                )

            # Check if this is a top-k model
            self._maybe_save_checkpoint(step, bypass_score, model)

    def _maybe_save_checkpoint(self, step: int, score: float, model):
        """Save checkpoint if it's in top-k."""
        if model is None:
            return

        checkpoint_path = self.checkpoints_dir / f"step_{step}"

        # If we have fewer than top_k, always save
        if len(self.top_models) < self.top_k:
            self._save_model(model, checkpoint_path)
            heapq.heappush(self.top_models, (score, step, str(checkpoint_path)))
            print(
                f"💾 Saved checkpoint (top {len(self.top_models)}) - step {step}, score={score:.4f}"
            )
        else:
            # Check if this score beats the worst in top-k
            worst_score, worst_step, worst_path = self.top_models[0]
            if score > worst_score:
                # Remove worst checkpoint
                heapq.heappop(self.top_models)
                if Path(worst_path).exists():
                    shutil.rmtree(worst_path)
                    print(
                        f"🗑️  Removed checkpoint step {worst_step} (score={worst_score:.4f})"
                    )

                # Save new checkpoint
                self._save_model(model, checkpoint_path)
                heapq.heappush(self.top_models, (score, step, str(checkpoint_path)))
                print(
                    f"💾 Saved checkpoint (top {self.top_k}) - step {step}, score={score:.4f}"
                )

    def _save_model(self, model, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_lora"):
            model.save_lora(str(path))
        else:
            model.save_pretrained(str(path))

    def print_summary(self):
        """Print bypass tracking summary."""
        if not self.step_metrics:
            return

        print(f"\n{'=' * 60}")
        print(f"📊 BYPASS TRACKING SUMMARY ({len(self.step_metrics)} steps)")
        print(f"{'=' * 60}")

        # Best step
        best_step = max(self.step_metrics, key=lambda x: x["bypass_score"])
        print(f"  Best step: {best_step['step']}")
        print(f"    Bypass score: {best_step['bypass_score']:.4f}")
        print(
            f"    Quality bypasses: {best_step['quality_bypass_count']}/{best_step['total_samples']} ({best_step['quality_bypass_rate']:.1%})"
        )
        print(f"    Avg judge score: {best_step['avg_quality_judge_score']:.2f}")

        # Top-k saved models
        print(f"\n  Top {self.top_k} saved checkpoints:")
        for score, step, path in sorted(self.top_models, reverse=True):
            print(f"    Step {step}: score={score:.4f} → {path}")

        print(f"{'=' * 60}\n")


class Profiler:
    """Simple profiler to track time spent in each component."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.token_counts = defaultdict(list)
        self.start_times = {}
        self.step_count = 0
        self.print_every = 1  # Log every step for detailed profiling
        self.last_reward_time = None  # Track time between reward calls

    def start(self, name: str):
        """Start timing a component."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.start_times[name] = time.perf_counter()

    def stop(self, name: str, num_tokens: int = 0):
        """Stop timing and record."""
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed)
            if num_tokens > 0:
                self.token_counts[name].append(num_tokens)
            del self.start_times[name]

    def record_generation_time(self, num_tokens: int):
        """Record time since last reward call (= generation time)."""
        now = time.perf_counter()
        if self.last_reward_time is not None:
            # Time between reward calls includes: generation + forward/backward
            # We'll track this as "step_total" (everything except reward)
            elapsed = now - self.last_reward_time
            # Subtract reward time to get generation + training time
            if self.timings.get("reward_total"):
                last_reward = (
                    self.timings["reward_total"][-1]
                    if self.timings["reward_total"]
                    else 0
                )
                gen_time = elapsed - last_reward
                if gen_time > 0:
                    self.timings["grpo_gen+train"].append(gen_time)
                    self.token_counts["grpo_gen+train"].append(num_tokens)
        self.last_reward_time = now

    def log_step(self, total_tokens: int = 0, total_samples: int = 0):
        """Log profiling info periodically."""
        self.step_count += 1

        # Record generation time (time since last reward call)
        self.record_generation_time(total_tokens)

        if self.step_count % self.print_every == 0:
            print(f"\n{'─' * 60}")
            print(
                f"⏱️  PROFILING (Step {self.step_count}, avg of last {self.print_every} steps)"
            )
            print(f"{'─' * 60}")

            component_times = {}

            for name in ["grpo_gen+train", "detector", "judge", "reward_total"]:
                times = self.timings.get(name, [])
                if times:
                    recent = (
                        times[-self.print_every :]
                        if len(times) >= self.print_every
                        else times
                    )
                    avg_time = sum(recent) / len(recent)
                    component_times[name] = avg_time

                    # Calculate tokens/sec if available
                    if name in self.token_counts and self.token_counts[name]:
                        recent_tokens = (
                            self.token_counts[name][-self.print_every :]
                            if len(self.token_counts[name]) >= self.print_every
                            else self.token_counts[name]
                        )
                        avg_tokens = sum(recent_tokens) / len(recent_tokens)
                        tps = avg_tokens / avg_time if avg_time > 0 else 0
                        print(
                            f"  {name:20s}: {avg_time:6.2f}s | {tps:,.0f} tok/s | {avg_tokens:,.0f} tokens"
                        )
                    else:
                        print(f"  {name:20s}: {avg_time:6.2f}s")

            # Calculate total step time
            if (
                "grpo_gen+train" in component_times
                and "reward_total" in component_times
            ):
                step_time = (
                    component_times["grpo_gen+train"] + component_times["reward_total"]
                )
                print(f"  {'─' * 40}")
                print(f"  {'TOTAL STEP':20s}: {step_time:6.2f}s")

                if total_samples > 0:
                    print(f"  {'Samples/step':20s}: {total_samples}")
                    print(f"  {'Samples/sec':20s}: {total_samples / step_time:.1f}")
                    print(
                        f"  {'Est. time/1000 samp':20s}: {1000 / total_samples * step_time / 60:.1f} min"
                    )

            print(f"{'─' * 60}\n")

            # Log to wandb if available
            if wandb.run is not None:
                wandb_data = {}
                for name, times in self.timings.items():
                    if times:
                        recent = (
                            times[-self.print_every :]
                            if len(times) >= self.print_every
                            else times
                        )
                        wandb_data[f"profiling/{name}_time"] = sum(recent) / len(recent)
                        if name in self.token_counts and self.token_counts[name]:
                            recent_tokens = self.token_counts[name][-self.print_every :]
                            avg_tokens = sum(recent_tokens) / len(recent_tokens)
                            avg_time = sum(recent) / len(recent)
                            wandb_data[f"profiling/{name}_tps"] = (
                                avg_tokens / avg_time if avg_time > 0 else 0
                            )
                if (
                    "grpo_gen+train" in component_times
                    and "reward_total" in component_times
                ):
                    step_time = (
                        component_times["grpo_gen+train"]
                        + component_times["reward_total"]
                    )
                    wandb_data["profiling/step_time"] = step_time
                    if total_samples > 0:
                        wandb_data["profiling/samples_per_sec"] = (
                            total_samples / step_time
                        )
                wandb.log(wandb_data, commit=False)


# Global profiler instance
profiler = Profiler()


def load_sft_model_for_rl(
    config: GRPOSpamConfig,
    sft_adapter_path: str,
):
    """Load SFT-trained model for RL fine-tuning using unsloth.

    Args:
        config: Training configuration
        sft_adapter_path: Path to SFT LoRA adapter

    Returns:
        Tuple of (model, tokenizer)
    """
    from unsloth import FastLanguageModel

    # Load base model with unsloth
    print(f"[1] Loading base model {config.generator_model} with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.generator_model,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.load_in_4bit,
    )

    # Load the SFT adapter weights manually
    print(f"[2] Loading SFT adapter from {sft_adapter_path}...")

    # Wrap with PEFT to load the SFT adapter
    model = PeftModel.from_pretrained(model, sft_adapter_path)

    # Merge SFT adapter into base weights
    print("[3] Merging SFT adapter into base model...")
    model = model.merge_and_unload()

    # Add fresh LoRA for RL training
    print("[4] Adding new LoRA adapter for RL...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model ready for RL training")
    return model, tokenizer


def create_simple_dataset(num_samples: int, seed: int = 42) -> Dataset:
    """Create dataset with simple prompts for RL training.

    The SFT model already knows how to generate spam,
    so we just need TASK_TAG to trigger generation.
    Note: enable_thinking=False is set in GRPOConfig via generation_config.
    """
    prompt = f"{TASK_TAG}"

    data = []
    for _ in range(num_samples):
        data.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "prompt_text": prompt,
            }
        )

    return Dataset.from_list(data)


class DetectorOnlyReward:
    """Simplified reward using only detector score.

    Reward = HAM probability - SPAM probability
    Range: [-1, +1]

    Logs bypass samples (HAM classification with spam payload) for analysis.
    """

    __name__ = "detector_only_reward"  # Required by GRPOTrainer

    def __init__(
        self,
        detector_path: str,
        device: str = "cuda",
        dtype: str = None,
        bypass_log_path: str = "bypass_samples.json",
        print_every: int = 10,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.print_every = print_every
        self.step_count = 0

        # Initialize bypass logger
        self.bypass_logger = BypassLogger(bypass_log_path)

        # Auto-detect dtype
        if dtype is None:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(0)
                if capability[0] >= 8:
                    dtype = "bfloat16"
                elif capability[0] >= 7 and capability[1] >= 5:
                    dtype = "float16"
                else:
                    dtype = "float32"
            else:
                dtype = "float32"

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        # Load detector
        DETECTOR_BASE_MODEL = "answerdotai/ModernBERT-base"
        self.tokenizer = AutoTokenizer.from_pretrained(DETECTOR_BASE_MODEL)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            DETECTOR_BASE_MODEL,
            num_labels=2,
            torch_dtype=torch_dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, detector_path).to(device)
        self.model.eval()

        print(f"  Detector loaded (dtype: {dtype})")
        print(f"  Bypass log: {bypass_log_path}")

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        """Compute detector-only reward.

        Returns HAM_prob - SPAM_prob for each completion.
        Also logs bypass samples (HAM + payload).
        """
        responses = [clean_response(c[0]["content"]) for c in completions]
        prompt_texts = [p[0]["content"] if p else "" for p in prompts]

        # Get detector predictions
        with torch.no_grad():
            inputs = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Reward = HAM_prob - SPAM_prob (range: -1 to +1)
        ham_probs = probs[:, 0].tolist()
        spam_probs = probs[:, 1].tolist()
        rewards = [(h - s) for h, s in zip(ham_probs, spam_probs)]

        # Log bypass samples (HAM classification + has spam payload)
        bypass_count_this_step = 0
        for i, (response, prompt, ham_prob, spam_prob, reward) in enumerate(
            zip(responses, prompt_texts, ham_probs, spam_probs, rewards)
        ):
            if ham_prob > 0.5 and has_spam_payload(response):
                self.bypass_logger.log_bypass(
                    sms=response,
                    ham_prob=ham_prob,
                    spam_prob=spam_prob,
                    reward=reward,
                    prompt=prompt,
                )
                bypass_count_this_step += 1

        # Print sample periodically
        if self.step_count % self.print_every == 0 and responses:
            bypass_rate = bypass_count_this_step / len(responses)
            print(f"\n{'=' * 50}")
            print(f"Step {self.step_count} | Sample Generated SMS:")
            msg = responses[0]
            print(f"SMS: {msg[:200]}..." if len(msg) > 200 else f"SMS: {msg}")
            print(f"Detector: HAM={ham_probs[0]:.2%}, SPAM={spam_probs[0]:.2%}")
            print(f"Has payload: {has_spam_payload(responses[0])}")
            print(f"Reward: {rewards[0]:.3f}")
            print(
                f"Bypass rate (this step): {bypass_rate:.1%} ({bypass_count_this_step}/{len(responses)})"
            )
            print(f"Total bypasses logged: {self.bypass_logger.get_count()}")
            print(f"{'=' * 50}")

        self.step_count += 1
        self.bypass_logger.step_count = self.step_count

        return rewards

    def finalize(self):
        """Finalize bypass log. Call at end of training."""
        self.bypass_logger.finalize()

    def get_bypass_count(self) -> int:
        """Return total bypass count."""
        return self.bypass_logger.get_count()


class LLMJudgeReward:
    """LLM-based judge to score SMS deceptiveness/harmfulness.

    Uses Qwen3-0.6B with disabled thinking to quickly evaluate
    how convincing/deceptive a spam SMS appears.

    Scoring:
    - 0: Not deceptive at all (obvious spam/garbage)
    - 1: Very deceptive (convincing scam)
    """

    __name__ = "llm_judge_reward"  # Required by GRPOTrainer

    JUDGE_SYSTEM_PROMPT = """You are a spam/scam SMS evaluator. Rate the following SMS message on how deceptive and convincing it is as a scam attempt.

Consider:
1. Believability - Does it seem like a legitimate message?
2. Urgency manipulation - Does it create false urgency effectively?
3. Professional appearance - Is the grammar/spelling convincing?
4. Call-to-action effectiveness - Is the payload (link/phone) presented naturally?

Respond with ONLY a single number from 0-10:
- 0-2: Obvious spam, poorly written, not convincing
- 3-4: Weak attempt, some red flags obvious
- 5-6: Moderate, could fool some people
- 7-8: Convincing, well-crafted deception
- 9-10: Highly deceptive, very professional scam"""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = None,
        weight: float = 0.3,  # Weight relative to detector reward
        print_every: int = 10,
    ):
        """Initialize LLM judge.

        Args:
            device: Device to run judge on
            dtype: Data type ("bfloat16", "float16", "float32")
            weight: Weight of judge score in total reward
            print_every: Print sample every N steps
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.weight = weight
        self.print_every = print_every
        self.step_count = 0

        # Auto-detect dtype
        if dtype is None:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(0)
                if capability[0] >= 8:
                    dtype = "bfloat16"
                elif capability[0] >= 7 and capability[1] >= 5:
                    dtype = "float16"
                else:
                    dtype = "float32"
            else:
                dtype = "float32"

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        # Load Qwen3-0.6B as judge
        print("  Loading LLM judge (Qwen3-0.6B)...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"  LLM judge loaded (dtype: {dtype}, weight: {weight})")

    def _score_sms(self, sms_list: list[str]) -> list[float]:
        """Score a batch of SMS messages.

        Returns scores normalized to [0, 1].
        """
        scores = []

        # Process in smaller batches to avoid OOM
        batch_size = 8
        for i in range(0, len(sms_list), batch_size):
            batch = sms_list[i : i + batch_size]
            batch_scores = self._score_batch(batch)
            scores.extend(batch_scores)

        return scores

    def _score_batch(self, sms_list: list[str]) -> list[float]:
        """Score a small batch of SMS messages."""
        scores = []

        for sms in sms_list:
            # Build prompt with /nothink to disable reasoning
            messages = [
                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Rate this SMS:\n\n{sms}\n\nScore (0-10):",
                },
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            # Add /nothink to disable thinking
            text = text.rstrip() + " /nothink\n"

            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # Just need a number
                    temperature=0.1,  # Low temp for consistent scoring
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            # Extract score from response
            score = self._parse_score(response)
            scores.append(score)

        return scores

    def _parse_score(self, response: str) -> float:
        """Parse numeric score from LLM response."""
        import re

        # Clean response
        response = response.strip()

        # Try to extract a number
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", response)
        if match:
            score = float(match.group(1))
            # Clamp to [0, 10] and normalize to [0, 1]
            score = max(0, min(10, score)) / 10.0
            return score

        # Default to middle score if parsing fails
        return 0.5

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        """Compute LLM judge reward.

        Returns weighted score for each completion.
        """
        responses = [clean_response(c[0]["content"]) for c in completions]

        # Get judge scores
        raw_scores = self._score_sms(responses)

        # Apply weight and shift to [-weight, +weight] range
        # Score of 0.5 (5/10) = neutral, higher = better
        rewards = [(s - 0.5) * 2 * self.weight for s in raw_scores]

        # Print sample periodically
        if self.step_count % self.print_every == 0 and responses:
            print(f"\n[LLM Judge] Step {self.step_count}")
            print(f"  SMS: {responses[0][:100]}...")
            print(f"  Raw score: {raw_scores[0]:.2f} (0-1)")
            print(f"  Weighted reward: {rewards[0]:.3f}")

        self.step_count += 1

        return rewards


def parse_judge_score(response: str) -> float:
    """Parse numeric score from LLM judge response.

    Robustly extracts a score from various response formats:
    - "7" -> 0.7
    - "Score: 8" -> 0.8
    - "I rate this 6/10" -> 0.6
    - "<think>...</think>5" -> 0.5

    Returns score normalized to [0, 1], or 0.5 if parsing fails.
    """
    import re

    # Clean response - remove think blocks if any leaked through
    response = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE
    )
    response = response.strip()

    # Try patterns in order of specificity
    patterns = [
        r"\b(\d+(?:\.\d+)?)\s*/\s*10\b",  # "7/10" format
        r"(?:score|rating|rate)[:\s]*(\d+(?:\.\d+)?)",  # "Score: 7" format
        r"^(\d+(?:\.\d+)?)$",  # Just a number
        r"\b(\d+(?:\.\d+)?)\b",  # Any number (fallback)
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = float(match.group(1))
            # Clamp to [0, 10] and normalize to [0, 1]
            score = max(0.0, min(10.0, score)) / 10.0
            return score

    # Default to middle score if parsing fails
    return 0.5


class CombinedReward:
    """Combined detector + LLM judge reward.

    Reward structure (judge-based anti-gaming):
    - DETECTOR_BYPASS_REWARD = +2.0 (if detector fooled AND judge score >= 0.3)
    - GAMING_PENALTY = -10.0 (if detector fooled BUT judge score < 0.3 = not real spam)
    - QUALITY_BONUS = up to +3.0 (judge_weight * judge_score for convincing spam)
    - DETECTED_PENALTY = -spam_prob (if detected as spam)

    The judge determines if a bypass is legitimate spam or just gaming.
    GAMING_THRESHOLD = 0.3 (3/10) - below this = not real spam

    Total reward = detector_reward + quality_bonus
    Max possible: +2.0 (bypass) + +3.0 (quality) = +5.0

    This rewards SMS that:
    1. Bypass the spam detector
    2. Are convincingly deceptive (judge score >= 0.3 for bypass, higher = more bonus)
    """

    __name__ = "combined_reward"  # Required by GRPOTrainer

    # Stricter judge prompt - requires actual spam characteristics
    JUDGE_PROMPT = """You are a strict spam/scam SMS evaluator. Rate ONLY based on these criteria:

1. MUST have a clear scam intent (fake prize, phishing, fraud, urgency trick)
2. MUST have a call-to-action (click link, call number, reply, send money)
3. Should create urgency or fear to manipulate the reader
4. Professional grammar/spelling makes it MORE deceptive (higher score)

A normal friendly message with NO scam intent = 0
A message about food/events with no fraud = 0-1  
Weak spam attempt, no clear CTA = 2-3
Basic spam with link/phone but obvious = 4-5
Convincing scam with urgency + CTA = 6-7
Highly deceptive professional scam = 8-10

SMS: {sms}

Score (0-10):"""

    def __init__(
        self,
        detector_path: str,
        device: str = "cuda",
        dtype: str = None,
        bypass_log_path: str = "bypass_samples.json",
        print_every: int = 10,
        judge_weight: float = 3.0,  # QUALITY_BONUS: +3 for high-quality spam
        use_judge: bool = True,
        diversity_weight: float = 1.0,  # Penalty for similar outputs in batch
    ):
        """Initialize combined reward.

        Args:
            detector_path: Path to spam detector
            device: Device to run on
            dtype: Data type
            bypass_log_path: Path to log bypass samples
            print_every: Print every N steps
            judge_weight: Weight for quality bonus (max +judge_weight for score=1.0)
            use_judge: Whether to use LLM judge for quality scoring
            diversity_weight: Penalty weight for similar outputs (anti-mode-collapse)
            judge_weight: Weight of LLM judge score (0 to disable)
            use_judge: Whether to use LLM judge
        """
        self.device = device
        self.print_every = print_every
        self.step_count = 0
        self.use_judge = use_judge and judge_weight > 0
        self.diversity_weight = diversity_weight

        # Initialize bypass logger
        self.bypass_logger = BypassLogger(bypass_log_path)

        # Auto-detect dtype
        if dtype is None:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(0)
                if capability[0] >= 8:
                    dtype = "bfloat16"
                elif capability[0] >= 7 and capability[1] >= 5:
                    dtype = "float16"
                else:
                    dtype = "float32"
            else:
                dtype = "float32"

        self.dtype = dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        # Load detector
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print("  Loading spam detector...")
        DETECTOR_BASE_MODEL = "answerdotai/ModernBERT-base"
        self.detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_BASE_MODEL)

        base_model = AutoModelForSequenceClassification.from_pretrained(
            DETECTOR_BASE_MODEL,
            num_labels=2,
            torch_dtype=torch_dtype,
        )
        self.detector = PeftModel.from_pretrained(base_model, detector_path).to(device)
        self.detector.eval()
        print(f"  Detector loaded (dtype: {dtype})")

        # Load LLM judge if enabled (using Unsloth for fast inference)
        # Uses Qwen3-4B as judge (smarter than generator, unfinetuned)
        self.judge = None
        self.judge_weight = judge_weight
        if self.use_judge:
            from unsloth import FastLanguageModel

            # Use Qwen3-4B-Instruct-2507 (non-thinking) as judge
            # Load in bf16 for faster inference (A100 has 80GB, 4B model is ~8GB)
            JUDGE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
            print(f"  Loading LLM judge ({JUDGE_MODEL})...")
            self.judge, self.judge_tokenizer = FastLanguageModel.from_pretrained(
                model_name=JUDGE_MODEL,
                max_seq_length=512,
                dtype=None,  # Auto-detect (bf16 on A100)
                load_in_4bit=False,  # bf16 is faster than 4-bit on A100
            )
            # Enable fast inference mode
            FastLanguageModel.for_inference(self.judge)

            if self.judge_tokenizer.pad_token is None:
                self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token

            print(f"  LLM judge loaded (bf16, weight: {judge_weight})")

        print(f"  Bypass log: {bypass_log_path}")

    def _get_detector_scores(
        self, responses: list[str]
    ) -> tuple[list[float], list[float]]:
        """Get detector HAM/SPAM probabilities."""
        profiler.start("detector")

        with torch.no_grad():
            inputs = self.detector_tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Count actual tokens (excluding padding) using attention_mask
            num_tokens = inputs.attention_mask.sum().item()
            outputs = self.detector(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        ham_probs = probs[:, 0].tolist()
        spam_probs = probs[:, 1].tolist()

        profiler.stop("detector", num_tokens)
        return ham_probs, spam_probs

    def _compute_diversity_penalties(self, responses: list[str]) -> list[float]:
        """Compute diversity penalty for each response based on similarity to others.

        For each response, compute average similarity to all other responses.
        High similarity = high penalty (encourages diverse outputs).

        Returns:
            List of penalties (0 = unique, up to diversity_weight = identical to others)
        """
        n = len(responses)
        if n <= 1:
            return [0.0] * n

        penalties = []
        for i, resp_i in enumerate(responses):
            # Compute similarity to all other responses
            similarities = []
            for j, resp_j in enumerate(responses):
                if i != j:
                    # Use SequenceMatcher for string similarity (0-1)
                    sim = SequenceMatcher(None, resp_i.lower(), resp_j.lower()).ratio()
                    similarities.append(sim)

            # Average similarity to others
            avg_sim = sum(similarities) / len(similarities) if similarities else 0

            # Penalty = similarity * weight
            # High similarity (>0.8) means near-duplicate, full penalty
            # Low similarity (<0.3) means diverse, no penalty
            penalty = (
                max(0, (avg_sim - 0.3) / 0.5) * self.diversity_weight
            )  # Scale 0.3-0.8 to 0-1
            penalties.append(penalty)

        return penalties

    def _get_judge_scores(self, responses: list[str]) -> list[float]:
        """Get LLM judge scores for responses using BATCHED Unsloth inference."""
        if not self.use_judge or self.judge is None:
            return [0.5] * len(responses)

        profiler.start("judge")
        total_generated_tokens = 0

        # Prepare all prompts
        all_texts = []
        for sms in responses:
            messages = [
                {"role": "user", "content": self.JUDGE_PROMPT.format(sms=sms[:300])},
            ]
            try:
                text = self.judge_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.judge_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            all_texts.append(text)

        # Batch process - use full batch size to maximize GPU utilization
        # On A100-80GB with 4B judge model, 256 batch fits easily
        BATCH_SIZE = 256  # Match generator batch size for single inference call
        all_scores = []

        for i in range(0, len(all_texts), BATCH_SIZE):
            batch_texts = all_texts[i : i + BATCH_SIZE]

            # Tokenize batch with padding
            inputs = self.judge_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.judge.generate(
                    **inputs,
                    max_new_tokens=8,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.judge_tokenizer.pad_token_id,
                )

            # Count generated tokens
            for j, output in enumerate(outputs):
                input_len = (
                    inputs.input_ids[j]
                    .ne(self.judge_tokenizer.pad_token_id)
                    .sum()
                    .item()
                )
                total_generated_tokens += len(output) - input_len

            # Decode each output and parse score
            for j, output in enumerate(outputs):
                # Get only the generated part (after input)
                input_len = (
                    inputs.input_ids[j]
                    .ne(self.judge_tokenizer.pad_token_id)
                    .sum()
                    .item()
                )
                generated = output[input_len:]
                response = self.judge_tokenizer.decode(
                    generated, skip_special_tokens=True
                )
                score = parse_judge_score(response)
                all_scores.append(score)

        profiler.stop("judge", total_generated_tokens)

        return all_scores

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        """Compute combined reward."""
        profiler.start("reward_total")

        responses = [clean_response(c[0]["content"]) for c in completions]
        prompt_texts = [p[0]["content"] if p else "" for p in prompts]

        # Count actual tokens in completions using detector tokenizer (consistent with detector counting)
        total_completion_tokens = sum(
            len(
                self.detector_tokenizer.encode(
                    c[0]["content"], add_special_tokens=False
                )
            )
            for c in completions
        )

        # Get detector scores
        ham_probs, spam_probs = self._get_detector_scores(responses)

        # Get judge scores FIRST (needed for reward calculation)
        if self.use_judge:
            judge_scores = self._get_judge_scores(responses)
        else:
            judge_scores = [0.5] * len(responses)

        # Reward structure (judge-based anti-gaming):
        # - DETECTOR_BYPASS_REWARD = +2.0 (flat reward for fooling detector)
        # - GAMING_PENALTY = -10.0 (if bypass but judge score < 0.3 = gaming attempt)
        # - QUALITY_BONUS = up to +3.0 (judge_weight * judge_score for convincing spam)
        # - DETECTED_PENALTY = -spam_prob (if detected as spam)
        #
        # The judge determines if the bypass is real spam or just gaming
        GAMING_THRESHOLD = 0.3  # Judge score below this = gaming (not real spam)

        detector_rewards = []
        for ham_prob, spam_prob, judge_score in zip(
            ham_probs, spam_probs, judge_scores
        ):
            if ham_prob > 0.5:  # Classified as HAM (bypass)
                if judge_score >= GAMING_THRESHOLD:
                    # Real bypass: detector fooled AND judge says it's spam
                    detector_rewards.append(2.0)
                else:
                    # Gaming: detector fooled BUT judge says it's not spam
                    detector_rewards.append(-10.0)
            else:
                # Detected as spam: small penalty proportional to confidence
                detector_rewards.append(-1.0 * spam_prob)

        # QUALITY_BONUS: +judge_weight * score for convincing spam
        # score=0 -> 0, score=0.5 -> +1.5, score=1.0 -> +3.0 (with default judge_weight=3.0)
        if self.use_judge:
            judge_rewards = [s * self.judge_weight for s in judge_scores]
        else:
            judge_rewards = [0.0] * len(responses)

        # Combined reward = detector + quality bonus
        total_rewards = [d + j for d, j in zip(detector_rewards, judge_rewards)]

        # Diversity penalty: penalize outputs too similar to others in batch
        # This prevents mode collapse where the model always generates the same SMS
        if self.diversity_weight > 0 and len(responses) > 1:
            diversity_penalties = self._compute_diversity_penalties(responses)
            total_rewards = [r - p for r, p in zip(total_rewards, diversity_penalties)]
        else:
            diversity_penalties = [0.0] * len(responses)

        # Track bypass metrics for callback
        bypass_count_this_step = 0
        quality_bypass_count = 0  # Bypasses with high judge score
        quality_judge_scores = []  # Judge scores for quality bypasses
        all_bypass_judge_scores = []  # Judge scores for all bypasses

        for i, (
            response,
            prompt,
            ham_prob,
            spam_prob,
            total_reward,
            judge_score,
        ) in enumerate(
            zip(
                responses,
                prompt_texts,
                ham_probs,
                spam_probs,
                total_rewards,
                judge_scores,
            )
        ):
            # Count detector bypass (ham_prob > 0.5)
            if ham_prob > 0.5:
                bypass_count_this_step += 1
                all_bypass_judge_scores.append(judge_score)

                # Quality bypass: detector bypass + high judge score (>= 0.5 = 5/10)
                if judge_score >= 0.5:
                    quality_bypass_count += 1
                    quality_judge_scores.append(judge_score)

                self.bypass_logger.log_bypass(
                    sms=response,
                    ham_prob=ham_prob,
                    spam_prob=spam_prob,
                    reward=total_reward,
                    prompt=prompt,
                    judge_score=judge_score if self.use_judge else None,
                )

        # Store step metrics for BypassTrackingCallback
        self.last_step_metrics = {
            "bypass_count": bypass_count_this_step,
            "quality_bypass_count": quality_bypass_count,
            "total_samples": len(responses),
            "avg_judge_score": sum(judge_scores) / len(judge_scores)
            if judge_scores
            else 0.5,
            "avg_quality_judge_score": sum(quality_judge_scores)
            / len(quality_judge_scores)
            if quality_judge_scores
            else 0.5,
        }

        # Print sample periodically
        if self.step_count % self.print_every == 0 and responses:
            bypass_rate = bypass_count_this_step / len(responses)
            quality_rate = quality_bypass_count / len(responses)
            print(f"\n{'=' * 60}")
            print(f"Step {self.step_count} | Combined Reward")
            msg = responses[0]
            print(f"SMS: {msg[:200]}..." if len(msg) > 200 else f"SMS: {msg}")
            print(
                f"Detector: HAM={ham_probs[0]:.2%}, SPAM={spam_probs[0]:.2%} → {detector_rewards[0]:.3f}"
            )
            if self.use_judge:
                print(f"Judge: {judge_scores[0]:.2f} → {judge_rewards[0]:.3f}")
            print(f"Total reward: {total_rewards[0]:.3f}")
            print(
                f"Bypass: {bypass_count_this_step}/{len(responses)} ({bypass_rate:.1%})"
            )
            print(
                f"Quality bypass (judge≥0.5): {quality_bypass_count}/{len(responses)} ({quality_rate:.1%})"
            )
            print(f"Total logged bypasses: {self.bypass_logger.get_count()}")
            print(f"{'=' * 60}")

        self.step_count += 1
        self.bypass_logger.step_count = self.step_count

        profiler.stop("reward_total")
        profiler.log_step(
            total_tokens=total_completion_tokens, total_samples=len(responses)
        )

        return total_rewards

    def finalize(self):
        """Finalize bypass log."""
        self.bypass_logger.finalize()

    def get_bypass_count(self) -> int:
        """Return total bypass count."""
        return self.bypass_logger.get_count()


def generate_samples(model, tokenizer, num_samples: int = 50) -> list[dict]:
    """Generate sample SMS messages from the trained model.

    Returns list of dicts with prompt, response, and metadata.
    """
    samples = []
    model.eval()

    prompt = f"{TASK_TAG}"

    for i in range(num_samples):
        messages = [{"role": "user", "content": prompt}]

        # Use enable_thinking=False in chat template to disable Qwen3 thinking
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,  # Disable Qwen3 thinking mode
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        samples.append(
            {
                "id": i,
                "prompt": prompt,
                "response": response,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    model.train()
    return samples


def train_rl_phase(
    config: GRPOSpamConfig,
    sft_adapter_path: str,
    output_dir: str = "rl_spam_lora",
    total_episodes: int = 4000,
    use_judge: bool = False,
    judge_weight: float = 0.6,
) -> tuple:
    """
    Phase 2: RL training with detector + optional LLM judge reward.

    Args:
        config: Training configuration
        sft_adapter_path: Path to SFT LoRA adapter
        output_dir: Where to save the RL LoRA adapter
        total_episodes: Total training episodes
        use_judge: Whether to use LLM judge (Qwen3-0.6B)
        judge_weight: Weight of judge score (0-1)

    Returns:
        Tuple of (model, tokenizer, trainer)
    """
    print("=" * 60)
    print(
        "RL Phase: "
        + ("Combined Detector + LLM Judge" if use_judge else "Detector-Only Reward")
    )
    print("=" * 60)

    # Print GPU config
    print("\n[GPU Configuration]")
    config.print_gpu_info()

    # Load SFT model
    print("\n[1/4] Loading SFT-trained model...")
    model, tokenizer = load_sft_model_for_rl(config, sft_adapter_path)

    # Load reward function
    print("\n[2/4] Loading reward function...")
    if use_judge:
        reward_func = CombinedReward(
            detector_path=config.detector_path,
            dtype=config.dtype,
            use_judge=True,
            judge_weight=judge_weight,
            diversity_weight=config.diversity_weight,
        )
    else:
        reward_func = DetectorOnlyReward(
            detector_path=config.detector_path,
            dtype=config.dtype,
        )

    # Create dataset
    print("\n[3/4] Creating prompt dataset...")
    dataset = create_simple_dataset(config.num_samples, config.seed)
    print(f"Created {len(dataset)} prompts")

    # Calculate max_steps
    effective_batch = config.per_device_batch_size * config.gradient_accumulation_steps
    max_steps = total_episodes // effective_batch
    print(f"Training for {max_steps} steps ({total_episodes} episodes)")

    # Configure GRPO
    print("\n[4/4] Starting RL training...")
    run_name = "rl-spam-combined" if use_judge else "rl-spam-detector-only"
    training_args = GRPOConfig(
        temperature=config.temperature,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=32,  # Short prompts
        max_completion_length=config.max_completion_length,
        max_steps=max_steps,
        num_train_epochs=1,  # Use max_steps to control training length
        save_steps=config.save_steps,
        report_to="wandb",
        output_dir=output_dir,
        run_name=run_name,
    )

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={
            "phase": "rl",
            "sft_adapter": sft_adapter_path,
            "total_episodes": total_episodes,
            "reward": "combined" if use_judge else "detector_only",
            "use_judge": use_judge,
            "judge_weight": judge_weight if use_judge else 0,
            **config.to_dict(),
        },
    )

    # Create profiling callback
    profiling_callback = GRPOProfilingCallback()

    # Create bypass tracking callback (saves top-2 models)
    bypass_callback = BypassTrackingCallback(
        reward_func=reward_func,
        output_dir=output_dir,
        top_k=2,
        judge_threshold=0.5,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],  # Single reward function
        args=training_args,
        train_dataset=dataset,
        callbacks=[profiling_callback, bypass_callback],
    )

    trainer.train()

    # Print final profiling summary
    profiling_callback.print_summary()
    bypass_callback.print_summary()

    # Finalize bypass log
    print("\n[POST] Finalizing bypass log...")
    reward_func.finalize()
    bypass_count = reward_func.get_bypass_count()
    print(f"  Total bypass samples: {bypass_count}")

    # Save
    print(f"\nSaving RL LoRA adapter to {output_dir}/")
    if hasattr(model, "save_lora"):
        model.save_lora(output_dir)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # Generate sample SMS messages
    print("\n[POST] Generating sample SMS messages...")
    generated_sms = generate_samples(model, tokenizer, num_samples=50)

    # Save generated SMS to file
    import json
    from pathlib import Path

    samples_path = Path(output_dir) / "generated_sms.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(generated_sms, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(generated_sms)} samples to {samples_path}")

    wandb.finish()

    print("\n✓ RL training complete!")
    print(f"  Adapter saved to: {output_dir}/")
    print(f"  Generated SMS: {samples_path}")
    print(f"  Bypass samples: bypass_samples.json ({bypass_count} samples)")

    return model, tokenizer, trainer
