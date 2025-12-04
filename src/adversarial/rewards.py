"""Reward functions for GRPO training."""

import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ModernBERT base model name
DETECTOR_BASE_MODEL = "answerdotai/ModernBERT-base"


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|info|biz)\b", re.IGNORECASE)

# Phone patterns - more flexible to match international formats
PHONE_PATTERN = re.compile(
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|"  # US: 123-456-7890, 1234567890
    r"\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b|"  # US: (123) 456-7890
    r"\b\+\d{1,3}[-.\s]?\d{6,12}\b|"  # International: +1 234567890, +44 123456789
    r"\b\d{5,6}\b"  # Short codes: 12345, 123456
, re.IGNORECASE)

# US format specifically (for realistic phone bonus)
US_PHONE_PATTERN = re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b")

# Short code / reply keyword patterns (common in SMS spam)
SHORT_CODE_PATTERN = re.compile(r"\b(text|reply|send|msg)\s+\w+\s+(to\s+)?\d{4,6}\b", re.IGNORECASE)
REPLY_KEYWORD_PATTERN = re.compile(r"\b(reply|text|send)\s+(stop|yes|no|help|info|start|join|quit)\b", re.IGNORECASE)

# Email pattern
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b", re.IGNORECASE)

# Call-to-action pattern (call us, dial now, etc.)
CALL_ACTION_PATTERN = re.compile(r"\b(call|dial|ring|contact)\s*(us|me|now|back|today|this)?\b", re.IGNORECASE)

URGENCY_WORDS = {"urgent", "immediately", "now", "asap", "expire", "limited", "today", "hours"}

# Qwen3 thinking tag (output even in /nothink mode)
THINK_TAG_PATTERN = re.compile(r"</think>\s*", re.IGNORECASE)


class RewardFunctions:
    """Reward functions for adversarial spam generation."""

    def __init__(
        self,
        detector_path: str,
        device: str = "cuda",
        print_every: int = 10,
        bypass_log_path: str = "bypass_samples.json",
        dtype: str = None,
    ):
        """
        Initialize reward functions with detector model.

        Args:
            detector_path: Path to fine-tuned ModernBERT spam classifier.
            device: Device to run detector on.
            print_every: Print sample every N steps.
            bypass_log_path: Path to save bypassing samples.
            dtype: Data type for model ("float32", "float16", "bfloat16"). Auto-detects if None.
        """
        self.device = device
        self.print_every = print_every
        self.step_count = 0
        self.bypass_log_path = Path(bypass_log_path)
        self.bypass_count = 0
        
        # Auto-detect dtype based on GPU if not specified
        if dtype is None:
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability(0)
                if capability[0] >= 8:  # Ampere+
                    dtype = "bfloat16"
                elif capability[0] >= 7 and capability[1] >= 5:  # Turing+
                    dtype = "float16"
                else:  # Pascal and older
                    dtype = "float32"
            else:
                dtype = "float32"
        
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(dtype, torch.float32)
        print(f"  Detector dtype: {dtype}")
        
        # Initialize/clear the bypass log file
        with open(self.bypass_log_path, "w") as f:
            f.write("[\n")  # Start JSON array

        # Load detector model (LoRA fine-tuned ModernBERT)
        # Tokenizer comes from the base model
        self.tokenizer = AutoTokenizer.from_pretrained(DETECTOR_BASE_MODEL)
        
        # Load base model then apply LoRA adapters
        base_model = AutoModelForSequenceClassification.from_pretrained(
            DETECTOR_BASE_MODEL,
            num_labels=2,
            torch_dtype=torch_dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, detector_path).to(device)
        self.model.eval()

        # Cache for detector predictions
        self._cache: dict[tuple, torch.Tensor] = {}
        
        # Cache for cleaned responses (avoid redundant cleaning across reward funcs)
        self._cleaned_cache: dict[int, str] = {}
        
        # Profiling stats
        self.detector_time_total = 0.0
        self.detector_calls = 0
        self.reward_time_total = 0.0
        self.reward_calls = 0

    def _get_detector_predictions(self, responses: list[str]) -> torch.Tensor:
        """
        Batch inference on detector with caching.

        Returns:
            Tensor of shape (batch_size, 2) with [ham_prob, spam_prob].
        """
        cache_key = tuple(responses)

        if cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.perf_counter()
        
        with torch.no_grad():
            inputs = self.tokenizer(
                responses,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        # Track detector time
        self.detector_time_total += time.perf_counter() - start_time
        self.detector_calls += 1

        # Cache and clear previous
        self._cache.clear()
        self._cache[cache_key] = probs

        return probs

    def detector_reward(self, prompts, completions, **kwargs) -> list[float]:
        """
        Main reward: +score if detector classifies as HAM, -score if SPAM.
        
        IMPORTANT: Only reward bypass if message contains spam payload (URL/phone/CTA).
        This prevents gaming with generic ham-like messages.

        Reward scaling:
        - HAM (bypass) with spam payload: +2.0 * ham_probability (max +2.0)
        - HAM (bypass) without payload: -0.5 (discourage generic messages)
        - SPAM (detected): -1.0 * spam_probability (max -1.0)
        """
        reward_start = time.perf_counter()
        
        # Clear cleaned cache at start of each step
        self._clear_cleaned_cache()
        
        # Clean responses: strip </think> tag from Qwen3 non-thinking mode output
        responses = [self._clean_response(c[0]["content"]) for c in completions]
        prompt_texts = [p[-1]["content"] for p in prompts]
        
        # Extract metadata if available (from dataset)
        metadata_list = kwargs.get("metadata", [None] * len(responses))
        
        probs = self._get_detector_predictions(responses)

        ham_probs = probs[:, 0]
        spam_probs = probs[:, 1]

        scores_list = []
        for i, (response, ham_prob, spam_prob) in enumerate(
            zip(responses, ham_probs.tolist(), spam_probs.tolist())
        ):
            has_payload = self._has_spam_payload(response)
            
            if ham_prob > 0.5:  # Classified as HAM
                if has_payload:
                    # Real bypass: HAM classification + spam payload
                    score = 4.0 * ham_prob
                else:
                    # Gaming: HAM but no payload (just a normal message)
                    score = -0.5
            else:
                # Detected as SPAM
                score = -1.0 * spam_prob
            
            scores_list.append(score)

        # Save bypassing samples progressively (only valid ones with payload)
        for i, (response, prompt_text, ham_prob, spam_prob, score) in enumerate(
            zip(responses, prompt_texts, ham_probs.tolist(), spam_probs.tolist(), scores_list)
        ):
            if ham_prob > 0.5 and self._has_spam_payload(response):  # Valid bypass
                # Get metadata for this sample
                meta = metadata_list[i] if i < len(metadata_list) else None
                
                sample = {
                    "step": self.step_count,
                    "sms": response,
                    "label": "spam",  # These are all spam by design
                    "prompt": prompt_text,
                    "ham_prob": round(ham_prob, 4),
                    "spam_prob": round(spam_prob, 4),
                    "detector_reward": round(score, 4),
                    "length": len(response),
                }
                
                # Add category info from metadata
                if meta and isinstance(meta, dict):
                    sample["objective"] = meta.get("objective", "unknown")
                    sample["context"] = meta.get("context", "unknown")
                    sample["requested_link"] = meta.get("has_link", False)
                    sample["requested_phone"] = meta.get("has_phone", False)
                    sample["requested_urgency"] = meta.get("has_urgency", False)
                else:
                    # Parse from prompt text as fallback
                    sample["objective"] = self._extract_objective(prompt_text)
                    sample["context"] = self._extract_context(prompt_text)
                
                self._append_bypass_sample(sample)
                self.bypass_count += 1

        # Track total reward computation time
        self.reward_time_total += time.perf_counter() - reward_start
        self.reward_calls += 1

        # Print examples periodically
        if self.step_count % self.print_every == 0:
            valid_bypass = sum(1 for r, hp in zip(responses, ham_probs.tolist()) 
                              if hp > 0.5 and self._has_spam_payload(r))
            bypass_rate = valid_bypass / len(responses)
            
            # Compute timing stats
            avg_detector_ms = (self.detector_time_total / max(self.detector_calls, 1)) * 1000
            avg_reward_ms = (self.reward_time_total / max(self.reward_calls, 1)) * 1000
            
            print(f"\n{'='*50}")
            print(f"Step {self.step_count} | Sample Generated SMS:")
            msg = responses[0]
            print(f"SMS: {msg[:200]}..." if len(msg) > 200 else f"SMS: {msg}")
            print(f"Detector: HAM={ham_probs[0]:.2%}, SPAM={spam_probs[0]:.2%}")
            print(f"Has payload: {self._has_spam_payload(responses[0])}")
            print(f"Reward: {scores_list[0]:.3f}")
            print(f"Valid bypass rate: {bypass_rate:.1%} ({valid_bypass}/{len(responses)})")
            print(f"Total valid bypasses logged: {self.bypass_count}")
            print(f"⏱️  Avg detector: {avg_detector_ms:.1f}ms | Avg reward total: {avg_reward_ms:.1f}ms")
            print(f"{'='*50}")

        self.step_count += 1
        return scores_list

    def _clean_response(self, text: str) -> str:
        """Clean response text from model artifacts (with caching).
        
        Removes:
        - </think> tag (Qwen3 outputs this even in /nothink mode)
        - Leading/trailing whitespace
        """
        # Use hash for cache key
        text_id = id(text)
        if text_id in self._cleaned_cache:
            return self._cleaned_cache[text_id]
        
        cleaned = THINK_TAG_PATTERN.sub("", text).strip()
        self._cleaned_cache[text_id] = cleaned
        return cleaned

    def _clear_cleaned_cache(self):
        """Clear the cleaned response cache (call at start of each step)."""
        self._cleaned_cache.clear()

    def _has_spam_payload(self, text: str) -> bool:
        """Check if message contains spam payload.
        
        A valid spam message needs a direct payload:
        - URL (link to malicious site), OR
        - Phone number (to call/text for scam), OR
        - Short code / reply keyword (text STOP to 12345), OR
        - Email address, OR
        - Call-to-action (call us, dial now)
        
        Also rejects garbage/noise outputs.
        """
        # Reject garbage outputs first
        if self._is_garbage(text):
            return False
        
        # URL patterns - direct payload (includes bare domains like example.com)
        has_url = URL_PATTERN.search(text) is not None
        
        # Phone patterns - any phone counts as payload (includes short codes)
        has_phone = PHONE_PATTERN.search(text) is not None
        
        # Short code patterns - "text STOP to 12345" or "reply YES"
        has_short_code = SHORT_CODE_PATTERN.search(text) is not None
        has_reply_keyword = REPLY_KEYWORD_PATTERN.search(text) is not None
        
        # Email address
        has_email = EMAIL_PATTERN.search(text) is not None
        
        # Call-to-action
        has_call_action = CALL_ACTION_PATTERN.search(text) is not None
        
        return has_url or has_phone or has_short_code or has_reply_keyword or has_email or has_call_action
    
    def _has_realistic_phone(self, text: str) -> bool:
        """Check if message contains a realistic (non-fake) US phone number.
        
        Used for bonus reward, not for payload validation.
        """
        # Only check US format phones for "realistic" bonus
        phone_match = US_PHONE_PATTERN.search(text)
        if not phone_match:
            return False
        return not self._is_fake_phone(phone_match.group())
    
    def _is_fake_phone(self, phone: str) -> bool:
        """Detect obviously fake phone numbers used for gaming.
        
        Catches:
        - Sequential digits (123-456-7890)
        - Repeated digits (555-555-5555)
        - Hollywood numbers (555-xxxx)
        - All same digit (000-000-0000)
        """
        # Extract just digits
        digits = re.sub(r'\D', '', phone)
        
        if len(digits) != 10:
            return True
        
        # Check for repeated digits (555-555-5555, 000-000-0000)
        if len(set(digits)) <= 2:
            return True
        
        # Check for sequential patterns (123-456-7890)
        if digits == "1234567890" or digits == "0987654321":
            return True
        
        # Check for 555 prefix (Hollywood fake numbers)
        if digits[3:6] == "555":
            return True
        
        # Check for repeated 3-digit patterns (123-123-1234)
        if digits[:3] == digits[3:6]:
            return True
        
        # Check for all same digit in area code or exchange
        if len(set(digits[:3])) == 1 or len(set(digits[3:6])) == 1:
            return True
        
        return False

    def _is_garbage(self, text: str) -> bool:
        """Detect garbage/noise outputs that aren't real messages.
        
        Catches:
        - Repetitive characters (e.g., "000000...")
        - Too few unique characters
        - Prompt leakage (contains instruction text)
        - Just phone numbers with no message
        - Random number prefixes
        """
        # Too short after stripping
        stripped = text.strip()
        if len(stripped) < 20:
            return True
        
        # Check for repetitive characters (more than 50% same char)
        if len(text) > 0:
            char_counts = {}
            for c in text:
                char_counts[c] = char_counts.get(c, 0) + 1
            max_char_ratio = max(char_counts.values()) / len(text)
            if max_char_ratio > 0.5:
                return True
        
        # Check for too few unique characters (garbage like "0000...")
        unique_chars = len(set(text.lower()))
        if unique_chars < 10 and len(text) > 30:
            return True
        
        # Check for random number prefix (e.g., "0\n", "160\n", "100000\n", "1234567890\n")
        lines = stripped.split('\n')
        first_line = lines[0].strip()
        if first_line.isdigit() and len(lines) > 1:
            return True
        
        # Note: Placeholders like [Name] are penalized in naturalness_reward, not rejected here
        
        # Check if message is ONLY phone numbers (no actual text content)
        # Remove phone numbers and see if anything meaningful is left
        text_without_phones = PHONE_PATTERN.sub('', text)
        text_without_phones = re.sub(r'[\s\-\n]+', ' ', text_without_phones).strip()
        if len(text_without_phones) < 15:  # Just phone numbers
            return True
        
        # Check for prompt leakage - expanded list
        prompt_indicators = [
            # Original
            "output only", "nothing else", "under 160 characters",
            "sound natural", "avoid spam", "/nothink", "</think>",
            "your message should",
            # New - catches meta-text
            "160 characters", "100 characters", "based on the instructions",
            "here's the input", "generate the", "your task is",
            "the user will", "sms message", "characters:",
            "the recipient to", "appears to be from",
            "\"note\"", "\"std\"",  # Weird token artifacts
            # Thinking artifacts
            "okay, let's", "let me ", "i'll ", "the user wants",
        ]
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in prompt_indicators):
            return True
        
        # Check for quoted instructions (model outputting what was asked)
        if text_lower.count('"') >= 4:  # Multiple quoted segments = likely meta
            return True
        
        return False

    def _extract_objective(self, prompt_text: str) -> str:
        """Extract objective from prompt text."""
        match = re.search(r"message's goal is: (.+?)\.?\n", prompt_text)
        return match.group(1) if match else "unknown"
    
    def _extract_context(self, prompt_text: str) -> str:
        """Extract context/source from prompt text."""
        match = re.search(r"appears to be from a (.+?)\.?\n", prompt_text)
        return match.group(1) if match else "unknown"

    def _append_bypass_sample(self, sample: dict):
        """Append a single bypass sample to the JSON file."""
        with open(self.bypass_log_path, "a") as f:
            prefix = "" if self.bypass_count == 0 else ",\n"
            f.write(f"{prefix}  {json.dumps(sample)}")
    
    def finalize_bypass_log(self):
        """Close the JSON array in the bypass log file."""
        with open(self.bypass_log_path, "a") as f:
            f.write("\n]")
        print(f"Finalized bypass log with {self.bypass_count} samples")
    
    def get_bypass_count(self) -> int:
        """Return total count of bypass samples."""
        return self.bypass_count

    def print_profiling_summary(self):
        """Print profiling summary of detector vs reward computation time."""
        print(f"\n{'='*60}")
        print("PROFILING SUMMARY")
        print(f"{'='*60}")
        print(f"Detector inference:")
        print(f"  - Total time: {self.detector_time_total:.2f}s")
        print(f"  - Calls: {self.detector_calls}")
        print(f"  - Avg per call: {(self.detector_time_total / max(self.detector_calls, 1)) * 1000:.1f}ms")
        print(f"Reward computation (total):")
        print(f"  - Total time: {self.reward_time_total:.2f}s")
        print(f"  - Calls: {self.reward_calls}")
        print(f"  - Avg per call: {(self.reward_time_total / max(self.reward_calls, 1)) * 1000:.1f}ms")
        
        # The difference is overhead (file I/O, regex, etc.)
        overhead = self.reward_time_total - self.detector_time_total
        detector_pct = (self.detector_time_total / max(self.reward_time_total, 0.001)) * 100
        print(f"Detector % of reward time: {detector_pct:.1f}%")
        print(f"Note: Generation time is tracked by GRPOTrainer (not shown here)")
        print(f"{'='*60}")

    def length_reward(self, prompts, completions, **kwargs) -> list[float]:
        """Reward for appropriate SMS length (bonus for 50-160 chars, penalty for too short/long)."""
        lengths = torch.tensor(
            [len(self._clean_response(c[0]["content"])) for c in completions], 
            dtype=torch.float32
        )

        # Penalize very short messages (gaming prevention)
        # Ideal: 30-160 chars, acceptable: up to 300, bad: <30 or >300
        scores = torch.where(
            lengths < 30,
            torch.tensor(-2.0),  # Heavy penalty for too short (prevents gaming)
            torch.where(
                lengths <= 160,
                torch.tensor(0.2),  # Slight bonus for SMS length
                torch.where(lengths <= 300, torch.tensor(0.0), torch.tensor(-0.3)),
            ),
        )

        return scores.tolist()

    def naturalness_reward(self, prompts, completions, **kwargs) -> list[float]:
        """Penalize template artifacts (placeholders) that indicate incomplete generation."""
        # Pre-compile placeholder pattern
        placeholder_pattern = re.compile(r'\[[A-Za-z]+\]')
        
        scores = []

        for completion in completions:
            text = self._clean_response(completion[0]["content"])
            penalty = 0.0
            
            # Penalize placeholder brackets like [Name], [Link], [Phone]
            # These indicate the model didn't fill in actual values
            placeholders = placeholder_pattern.findall(text)
            if placeholders:
                penalty += 0.3 * len(placeholders)  # -0.3 per placeholder
            
            scores.append(-min(penalty, 1.5))  # Cap at -1.5

        return scores

    def task_completion_reward(self, prompts, completions, **kwargs) -> list[float]:
        """Reward for including spam payload elements (links, phones, etc.)."""
        scores = []

        for prompt, completion in zip(prompts, completions):
            text = self._clean_response(completion[0]["content"])
            text_lower = text.lower()
            prompt_text = prompt[-1]["content"].lower()
            score = 0.0

            # Base reward for having ANY payload (the core spam signal)
            if self._has_spam_payload(text):
                score += 0.5  # Main payload bonus
            
            # Extra bonus for specific requested elements
            if "url" in prompt_text or "link" in prompt_text:
                if URL_PATTERN.search(text):
                    score += 0.2

            if "phone" in prompt_text:
                if PHONE_PATTERN.search(text):
                    score += 0.2
                    # Bonus for realistic phone number
                    if self._has_realistic_phone(text):
                        score += 0.15

            # Bonus for realistic phone even if not explicitly requested
            elif self._has_realistic_phone(text):
                score += 0.1

            # Check if prompt asked for urgency
            if "urgency" in prompt_text:
                text_words = set(text_lower.split())
                if text_words & URGENCY_WORDS:
                    score += 0.15

            scores.append(score)

        return scores

    def get_all_reward_funcs(self) -> list:
        """Return list of all reward functions for GRPOTrainer."""
        return [
            self.detector_reward,
            self.length_reward,
            self.naturalness_reward,
            self.task_completion_reward,
        ]
