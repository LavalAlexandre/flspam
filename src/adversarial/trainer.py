"""GRPO trainer for adversarial spam generation."""

# Import unsloth FIRST to ensure all optimizations are applied
from unsloth import FastLanguageModel

import json
import random
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from .config import GRPOSpamConfig
from .prompts import CHAT_TEMPLATE, SYSTEM_PROMPT, create_prompt_messages, generate_spam_prompt
from .rewards import RewardFunctions


def load_generator_model(config: GRPOSpamConfig):
    """Load and configure the generator model with LoRA."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.generator_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        fast_inference=False,  # Disable vLLM - incompatible with Colab's CUDA init
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.gpu_memory_utilization,
        dtype=config.torch_dtype,  # Use GPU-appropriate dtype
    )

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
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    tokenizer.chat_template = CHAT_TEMPLATE

    return model, tokenizer


def create_dataset(config: GRPOSpamConfig) -> Dataset:
    """Generate dataset of spam task prompts with metadata."""
    random.seed(config.seed)

    data = []
    for _ in range(config.num_samples):
        prompt_text, metadata = generate_spam_prompt()
        data.append({
            "prompt": create_prompt_messages(prompt_text),
            "prompt_text": prompt_text,  # Raw prompt for logging
            "metadata": metadata,  # Category info for bypass samples
            "answer": "",
        })

    return Dataset.from_list(data)


def train_adversarial_generator(config: GRPOSpamConfig) -> tuple:
    """
    Train adversarial spam generator using GRPO.

    Args:
        config: Training configuration.

    Returns:
        Tuple of (model, tokenizer, trainer).
    """
    print("=" * 60)
    print("GRPO Adversarial Spam Generator Training")
    print("=" * 60)
    
    # Print GPU configuration
    print("\n[GPU Configuration]")
    config.print_gpu_info()

    # Load models
    print("\n[1/5] Loading generator model...")
    model, tokenizer = load_generator_model(config)

    print("\n[2/5] Loading detector model...")
    reward_funcs = RewardFunctions(
        detector_path=config.detector_path,
        print_every=config.print_every,
        bypass_log_path=config.bypass_log_path,
        dtype=config.dtype,  # Use same dtype as generator
    )

    print("\n[3/5] Creating dataset...")
    dataset = create_dataset(config)
    print(f"Created {len(dataset)} training samples")

    # Configure GRPO
    print("\n[4/5] Configuring GRPO trainer...")

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
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        num_train_epochs=1,  # Use max_steps to control training length
        save_steps=config.save_steps,
        report_to="wandb",
        output_dir=config.output_dir,
        run_name=config.wandb_run_name,
    )

    # Initialize W&B
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.to_dict(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs.get_all_reward_funcs(),
        args=training_args,
        train_dataset=dataset,
    )

    # Train
    print("\n[5/5] Starting training...")
    print("Watch the 'reward' column - it should increase as the model learns!")
    trainer.train()

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {config.lora_output_dir}/")
    # Use PEFT's save_pretrained (works with both unsloth and standard PEFT models)
    if hasattr(model, 'save_lora'):
        # Unsloth model
        model.save_lora(config.lora_output_dir)
    else:
        # Standard PEFT model (after GRPOTrainer)
        model.save_pretrained(config.lora_output_dir)
        tokenizer.save_pretrained(config.lora_output_dir)

    # Finalize bypass log and print profiling
    reward_funcs.finalize_bypass_log()
    reward_funcs.print_profiling_summary()
    print(f"Saved {reward_funcs.get_bypass_count()} bypass samples to {config.bypass_log_path}")

    return model, tokenizer, trainer


def generate_samples(
    model,
    tokenizer,
    lora_path: str,
    prompt_text: str,
    num_samples: int = 5,
) -> list[str]:
    """Generate adversarial spam samples using standard HuggingFace generation."""
    messages = create_prompt_messages(prompt_text)

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate using standard HuggingFace
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract generated text (remove input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_texts = []
    for output in outputs:
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
        generated_texts.append(generated.strip())
    
    return generated_texts


def evaluate_bypass_rate(
    model,
    tokenizer,
    reward_funcs: RewardFunctions,
    lora_path: str,
    num_eval: int = 100,
) -> dict:
    """Evaluate bypass rate of trained model."""
    print(f"\nEvaluating on {num_eval} samples...")

    bypass_count = 0
    total_ham_prob = 0.0

    random.seed(123)
    for i in range(num_eval):
        prompt = generate_spam_prompt()
        samples = generate_samples(model, tokenizer, lora_path, prompt, num_samples=1)

        if samples:
            sample = samples[0]
            probs = reward_funcs._get_detector_predictions([sample])
            ham_prob = probs[0, 0].item()
            total_ham_prob += ham_prob

            if ham_prob > 0.5:
                bypass_count += 1

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{num_eval}...")

    results = {
        "bypass_rate": bypass_count / num_eval,
        "avg_ham_prob": total_ham_prob / num_eval,
        "num_eval": num_eval,
    }

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS ({num_eval} samples)")
    print(f"{'='*50}")
    print(f"Bypass Rate: {results['bypass_rate']:.1%}")
    print(f"Average HAM Probability: {results['avg_ham_prob']:.1%}")

    wandb.log({
        "eval/bypass_rate": results["bypass_rate"],
        "eval/avg_ham_prob": results["avg_ham_prob"],
    })

    return results


def export_adversarial_samples(
    model,
    tokenizer,
    reward_funcs: RewardFunctions,
    lora_path: str,
    config: GRPOSpamConfig,
) -> list[dict]:
    """Export high-quality adversarial samples for FL training."""
    print(f"\nExporting {config.export_num_samples} adversarial samples...")
    print(f"Bypass threshold: {config.export_bypass_threshold:.0%}")

    adversarial_samples = []
    attempts = 0
    max_attempts = config.export_num_samples * 3

    random.seed(456)
    while len(adversarial_samples) < config.export_num_samples and attempts < max_attempts:
        prompt = generate_spam_prompt()
        samples = generate_samples(model, tokenizer, lora_path, prompt, num_samples=1)
        attempts += 1

        if samples:
            sample = samples[0]
            probs = reward_funcs._get_detector_predictions([sample])
            ham_prob = probs[0, 0].item()

            if ham_prob >= config.export_bypass_threshold:
                adversarial_samples.append({
                    "text": sample,
                    "bypass_confidence": ham_prob,
                    "prompt": prompt,
                })

        if len(adversarial_samples) % 50 == 0 and len(adversarial_samples) > 0:
            print(f"  Collected {len(adversarial_samples)}/{config.export_num_samples} (attempts: {attempts})")

    # Save to file
    output_path = Path(config.export_path)
    with open(output_path, "w") as f:
        json.dump(adversarial_samples, f, indent=2)

    print(f"\nSaved {len(adversarial_samples)} adversarial samples to {output_path}")
    print(f"Acceptance rate: {len(adversarial_samples)/attempts:.1%}")

    return adversarial_samples
