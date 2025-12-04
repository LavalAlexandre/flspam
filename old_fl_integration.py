"""Integration of adversarial training into the Federated Learning pipeline."""

import gc
import json
import random
from pathlib import Path

import torch


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def run_adversarial_training(
    detector_path: str,
    output_dir: str = "adversarial_round",
    total_episodes: int = 4000,  # Reduced for FL integration
    batch_size: int = 64,
) -> str:
    """
    Run adversarial training against the current detector.

    Args:
        detector_path: Path to the current aggregated detector model.
        output_dir: Directory to save adversarial outputs.
        total_episodes: Total training episodes (reduced for speed in FL).
        batch_size: Batch size for training.

    Returns:
        Path to the bypass_samples.json file.
    """
    print("\n" + "=" * 60)
    print("ADVERSARIAL TRAINING PHASE")
    print("=" * 60)

    # Clear memory before loading adversarial models
    clear_memory()
    print("[ADV] Cleared memory before adversarial training")

    # Import here to avoid loading heavy models at module import time
    from .config import GRPOSpamConfig
    from .trainer import train_adversarial_generator

    # Create config for this adversarial round
    bypass_log_path = f"{output_dir}/bypass_samples.json"

    config = GRPOSpamConfig(
        detector_path=detector_path,
        output_dir=f"{output_dir}/grpo_outputs",
        lora_output_dir=f"{output_dir}/grpo_lora",
        bypass_log_path=bypass_log_path,
        total_episodes=total_episodes,
        per_device_batch_size=batch_size,
        num_samples=total_episodes,  # Match samples to episodes
        wandb_project="flspam-adversarial",
        wandb_run_name=f"adversarial-{Path(detector_path).name}",
        print_every=5,  # More frequent logging for shorter runs
    )

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Train adversarial generator
        model, tokenizer, trainer = train_adversarial_generator(config)

        # Cleanup adversarial models
        del model, tokenizer, trainer

    except Exception as e:
        print(f"[ADV] Error during training: {e}")
        raise
    finally:
        # Always clear memory after adversarial training
        clear_memory()
        print("[ADV] Cleared memory after adversarial training")

    return bypass_log_path


def select_best_bypasses(
    bypass_log_path: str,
    num_samples: int = 200,
    min_ham_prob: float = 0.6,
) -> list[dict]:
    """
    Select the best bypass samples from the log.

    Args:
        bypass_log_path: Path to bypass_samples.json.
        num_samples: Maximum number of samples to select.
        min_ham_prob: Minimum HAM probability threshold.

    Returns:
        List of selected spam samples.
    """
    print(f"\n[ADV] Selecting best bypasses from {bypass_log_path}")

    # Load bypass samples
    with open(bypass_log_path, "r") as f:
        content = f.read()
        # Handle incomplete JSON (missing closing bracket)
        if not content.strip().endswith("]"):
            content = content.rstrip().rstrip(",") + "\n]"
        bypasses = json.loads(content)

    print(f"[ADV] Loaded {len(bypasses)} raw bypass samples")

    # Filter by HAM probability
    filtered = [s for s in bypasses if s.get("ham_prob", 0) >= min_ham_prob]
    print(f"[ADV] {len(filtered)} samples above {min_ham_prob:.0%} threshold")

    # Sort by HAM probability (best bypasses first)
    filtered.sort(key=lambda x: x.get("ham_prob", 0), reverse=True)

    # Deduplicate by SMS text (keep highest ham_prob version)
    seen_texts = set()
    unique = []
    for sample in filtered:
        text = sample.get("sms", "").strip().lower()
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique.append(sample)

    print(f"[ADV] {len(unique)} unique samples after deduplication")

    # Select top N
    selected = unique[:num_samples]
    print(f"[ADV] Selected {len(selected)} best bypass samples")

    return selected


def add_adversarial_to_dataset(
    selected_samples: list[dict],
    spam_file: str = "data/spam_messages.json",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Add adversarial samples to the spam dataset.

    Args:
        selected_samples: List of bypass samples to add.
        spam_file: Path to spam_messages.json.
        train_ratio: Fraction for training (rest goes to validation).
        seed: Random seed for train/val split.

    Returns:
        Dict with stats about added samples.
    """
    print(f"\n[ADV] Adding {len(selected_samples)} adversarial samples to dataset")

    # Load existing spam
    with open(spam_file, "r") as f:
        spam_data = json.load(f)

    original_count = len(spam_data)
    print(f"[ADV] Original spam count: {original_count}")

    # Convert bypass samples to spam format
    random.seed(seed)
    random.shuffle(selected_samples)

    train_count = int(len(selected_samples) * train_ratio)

    added_train = 0
    added_val = 0

    for i, sample in enumerate(selected_samples):
        is_train = i < train_count

        spam_entry = {
            "text": sample["sms"],
            "label": 1,  # Spam
            "source": "adversarial",
            "ham_prob": sample.get("ham_prob", 0),
            "objective": sample.get("objective", "unknown"),
            "context": sample.get("context", "unknown"),
            "split": "train" if is_train else "val",
        }

        spam_data.append(spam_entry)

        if is_train:
            added_train += 1
        else:
            added_val += 1

    # Save updated spam file
    with open(spam_file, "w") as f:
        json.dump(spam_data, f, indent=2)

    stats = {
        "original_count": original_count,
        "added_train": added_train,
        "added_val": added_val,
        "added_total": added_train + added_val,
        "new_total": len(spam_data),
    }

    print(f"[ADV] Added {added_train} train + {added_val} val samples")
    print(f"[ADV] New spam total: {len(spam_data)}")

    return stats


def run_adversarial_round(
    detector_path: str,
    round_num: int,
    output_base_dir: str = "adversarial_outputs",
    num_samples_to_add: int = 200,
    total_episodes: int = 4000,
    batch_size: int = 64,
    min_ham_prob: float = 0.6,
) -> dict:
    """
    Complete adversarial round: train generator, select best samples, add to dataset.

    Args:
        detector_path: Path to current detector model.
        round_num: FL round number (for logging).
        output_base_dir: Base directory for outputs.
        num_samples_to_add: How many samples to add to dataset.
        total_episodes: Training episodes for adversarial generator.
        batch_size: Batch size.
        min_ham_prob: Minimum bypass confidence.

    Returns:
        Dict with statistics.
    """
    output_dir = f"{output_base_dir}/round_{round_num}"

    print(f"\n{'#' * 60}")
    print(f"# ADVERSARIAL ROUND {round_num}")
    print(f"# Detector: {detector_path}")
    print(f"# Output: {output_dir}")
    print(f"{'#' * 60}")

    # Step 1: Train adversarial generator
    bypass_log_path = run_adversarial_training(
        detector_path=detector_path,
        output_dir=output_dir,
        total_episodes=total_episodes,
        batch_size=batch_size,
    )

    # Step 2: Select best bypass samples
    selected = select_best_bypasses(
        bypass_log_path=bypass_log_path,
        num_samples=num_samples_to_add,
        min_ham_prob=min_ham_prob,
    )

    # Step 3: Add to dataset
    dataset_stats = add_adversarial_to_dataset(
        selected_samples=selected,
        train_ratio=0.8,
    )

    # Combined stats
    stats = {
        "round": round_num,
        "detector_path": detector_path,
        "bypass_log_path": bypass_log_path,
        "num_bypasses_found": len(selected),
        **dataset_stats,
    }

    # Save stats
    stats_path = f"{output_dir}/adversarial_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[ADV] Round {round_num} complete! Stats saved to {stats_path}")

    return stats
