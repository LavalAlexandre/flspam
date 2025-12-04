"""SFT (Supervised Fine-Tuning) on spam dataset.

Phase 1 of the SFT → RL pipeline:
Train the generator to produce spam messages by learning from real spam data.
"""

import json
import random

from datasets import Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel

from .bypass_logger import TASK_TAG
from .config import GRPOSpamConfig


class SFTSampleCallback(TrainerCallback):
    """Callback to log progress during SFT training (no generation to avoid OOM)."""

    def __init__(self, tokenizer, log_every_steps: int = 100):
        self.tokenizer = tokenizer
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Just log progress, no generation (causes OOM with unsloth)
        if state.global_step % self.log_every_steps == 0:
            print(
                f"\n[SFT] Step {state.global_step}/{state.max_steps} | Loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}"
            )

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n{'=' * 50}")
        print(f"[SFT] Epoch {state.epoch:.1f} complete")
        print(f"{'=' * 50}\n")


def load_spam_dataset(
    spam_path: str, tokenizer, seed: int = 42, max_length: int = 400
) -> Dataset:
    """Load spam messages from JSON file.

    Args:
        spam_path: Path to spam_messages.json
        tokenizer: Tokenizer to format with chat template
        seed: Random seed for shuffling
        max_length: Maximum character length (to avoid tokenization issues)

    Returns:
        Dataset with 'text' column containing formatted spam messages
    """
    with open(spam_path, "r") as f:
        spam_data = json.load(f)

    # Extract text, filter by length (too short or too long)
    raw_texts = [
        msg["text"] for msg in spam_data if 20 <= len(msg["text"]) <= max_length
    ]

    # Format as chat - Qwen3-4B-Instruct-2507 is non-thinking by default
    # We still use enable_thinking=False for compatibility with other models
    texts = []
    for spam_text in raw_texts:
        messages = [
            {"role": "user", "content": TASK_TAG},
            {"role": "assistant", "content": spam_text},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,  # Not needed for 2507 but kept for compatibility
            )
        except TypeError:
            # Fallback if tokenizer doesn't support enable_thinking
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        texts.append(formatted)

    # Shuffle
    random.seed(seed)
    random.shuffle(texts)

    print(f"Loaded {len(texts)} spam messages (filtered to {max_length} chars max)")
    print(f"  Format: chat template with '{TASK_TAG}'")

    return Dataset.from_dict({"text": texts})


def format_for_sft(example: dict) -> dict:
    """Format spam message for SFT training.

    Simple format: just the spam text with EOS token.
    No system prompt, no instruction - pure text completion.
    """
    return {"text": example["text"]}


def load_sft_model(config: GRPOSpamConfig):
    """Load model for SFT training."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.generator_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=config.torch_dtype,
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

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def train_sft(
    config: GRPOSpamConfig,
    spam_path: str = "data/spam_messages.json",
    output_dir: str = "sft_spam_lora",
    num_epochs: int = 3,
    max_samples: int = None,
) -> tuple:
    """
    Train SFT on spam dataset.

    Args:
        config: Training configuration
        spam_path: Path to spam messages JSON
        output_dir: Where to save the LoRA adapter
        num_epochs: Number of training epochs
        max_samples: Max samples to use (None = all)

    Returns:
        Tuple of (model, tokenizer, trainer)
    """
    print("=" * 60)
    print("SFT Training on Spam Dataset")
    print("=" * 60)

    # Print GPU config
    print("\n[GPU Configuration]")
    config.print_gpu_info()

    # Load model
    print("\n[1/3] Loading model...")
    model, tokenizer = load_sft_model(config)

    # Load dataset
    print("\n[2/3] Loading spam dataset...")
    dataset = load_spam_dataset(spam_path, tokenizer, config.seed)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples")

    # Training args
    print("\n[3/3] Starting SFT training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate * 2,  # Slightly higher LR for SFT
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_8bit",
        fp16=config.dtype == "float16",
        bf16=config.dtype == "bfloat16",
        report_to="wandb",
        run_name="sft-spam-generator",
    )

    # Create sample logging callback
    sample_callback = SFTSampleCallback(tokenizer, log_every_steps=50)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",  # Explicitly specify text field
        packing=False,  # Disable packing to avoid length issues
        callbacks=[sample_callback],
    )

    trainer.train()

    # Save
    print(f"\nSaving SFT LoRA adapter to {output_dir}/")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n✓ SFT training complete!")
    print(f"  Adapter saved to: {output_dir}/")

    return model, tokenizer, trainer


def generate_from_sft(
    model,
    tokenizer,
    num_samples: int = 5,
    max_length: int = 160,
    temperature: float = 0.8,
) -> list[str]:
    """Generate spam samples from SFT model.

    Args:
        model: Trained SFT model
        tokenizer: Tokenizer
        num_samples: Number of samples to generate
        max_length: Max generation length
        temperature: Sampling temperature

    Returns:
        List of generated spam messages
    """
    FastLanguageModel.for_inference(model)

    prompt = f"{TASK_TAG}"

    samples = []
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

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Extract only the generated response (remove input prompt)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        samples.append(response.strip())

    return samples
