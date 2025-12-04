"""Integration of adversarial training into the Federated Learning pipeline."""

import gc
import json
import random
from pathlib import Path
from typing import Literal

import torch
import wandb

from ..config_defaults import (
    DEFAULT_ADVERSARIAL_EPISODES,
    DEFAULT_ADVERSARIAL_SAMPLES,
    DEFAULT_SFT_EPOCHS,
)


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
    total_episodes: int = DEFAULT_ADVERSARIAL_EPISODES,
    batch_size: int = 64,
    wandb_run_id: str = None,
    sft_epochs: int = DEFAULT_SFT_EPOCHS,
) -> str:
    """
    Run adversarial training against the current detector.
    
    Args:
        detector_path: Path to the current aggregated detector model.
        output_dir: Directory to save adversarial outputs.
        total_episodes: Total training episodes (reduced for speed in FL).
        batch_size: Batch size for training.
        wandb_run_id: ID of the active FL W&B run (to resume after).
        sft_epochs: Number of epochs for SFT training.
        
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
    from .sft import train_sft
    from .rl_simple import train_rl_phase

    
    # Create config for this adversarial round
    # Use absolute path to ensure file is created in correct location
    output_dir_path = Path(output_dir).resolve()
    bypass_log_path = str(output_dir_path / "bypass_samples.json")
    
    config = GRPOSpamConfig(
        detector_path=detector_path,
        output_dir=str(output_dir_path / "grpo_outputs"),
        lora_output_dir=str(output_dir_path / "grpo_lora"),
        bypass_log_path=bypass_log_path,
        total_episodes=total_episodes,
        per_device_batch_size=batch_size,
        num_samples=total_episodes,  # Match samples to episodes
        wandb_project="flspam-adversarial",
        wandb_run_name=f"adversarial-{Path(detector_path).name}",
        print_every=5,  # More frequent logging for shorter runs
    )
    
    # Create output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # If we have an active FL run, finish it so adversarial training can start its own run
    if wandb_run_id:
        print(f"[ADV] Pausing FL W&B run ({wandb_run_id}) for adversarial training...")
        wandb.finish()
    
    try:
        # Phase 1: SFT (Supervised Fine-Tuning)
        # Check if SFT adapter exists, if not train it
        sft_output_dir = "sft_spam_lora"
        if not Path(sft_output_dir).exists():
            print("\n" + "=" * 60)
            print("PHASE 1: SFT (First Run Only)")
            print("=" * 60)
            try:
                # Train SFT model
                model, tokenizer, trainer = train_sft(
                    config=config,
                    output_dir=sft_output_dir,
                    num_epochs=sft_epochs,  # Use configured epochs
                )
                # Cleanup SFT model
                del model, tokenizer, trainer
                clear_memory()
                print("[ADV] SFT phase complete and memory cleared")
            except Exception as e:
                print(f"[ADV] Error during SFT training: {e}")
                raise

        # Phase 2: RL (GRPO)
        # Always train RL fresh from SFT model (no resuming from previous RL)
        print("\n" + "=" * 60)
        print("PHASE 2: RL (Adversarial Optimization)")
        print("=" * 60)
        
        # Clean RL output directory to ensure fresh training each time
        rl_output_dir = output_dir_path / "rl_adapter"
        if rl_output_dir.exists():
            import shutil
            print(f"[ADV] Removing old RL adapter at {rl_output_dir} to train fresh from SFT...")
            shutil.rmtree(rl_output_dir)
        
        try:
            # Train RL phase using SFT adapter as starting point
            # Note: load_sft_model_for_rl always merges SFT and adds fresh LoRA
            train_rl_phase(
                config=config,
                sft_adapter_path=sft_output_dir,
                output_dir=str(rl_output_dir),
                total_episodes=total_episodes,
                use_judge=True,  # Use judge to detect gaming
                judge_weight=3.0,  # QUALITY_BONUS: +3 for high-quality spam
                bypass_log_path=bypass_log_path,
            )
            
        except Exception as e:
            print(f"[ADV] Error during RL training: {e}")
            raise
        finally:
            # Always clear memory after adversarial training
            clear_memory()
            print("[ADV] Cleared memory after adversarial training")
        
        return bypass_log_path
        
    except Exception as e:
        print(f"[ADV] Error during training: {e}")
        raise
    finally:
        # Always clear memory after adversarial training
        clear_memory()
        print("[ADV] Cleared memory after adversarial training")
        
        # Resume FL W&B run if it existed
        if wandb_run_id:
            print(f"[ADV] Resuming FL W&B run ({wandb_run_id})...")
            wandb.init(id=wandb_run_id, resume="allow", project="flspam")
    
    return bypass_log_path


def select_best_bypasses(
    bypass_log_path: str,
    num_samples: int = 200,
    min_ham_prob: float = 0.5,
    min_judge_score: float = 0.5,
    round_num: int = 0,
) -> list[dict]:
    """
    Select the best bypass samples from the log.
    Args:
        bypass_log_path: Path to bypass_samples.json.
        num_samples: Maximum number of samples to select.
        min_ham_prob: Minimum HAM probability threshold.
        min_judge_score: Minimum judge score threshold.
        round_num: FL round number (for filename).
        
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
    
    # Filter by HAM probability and judge score
    filtered = [s for s in bypasses if s.get("ham_prob", 0) >= min_ham_prob and s.get("judge_score", 0) >= min_judge_score]
    print(f"[ADV] {len(filtered)} samples above {min_ham_prob:.0%} HAM and {min_judge_score:.1f} judge score thresholds")
    # Sort by HAM probability (best bypasses first)
    filtered.sort(key=lambda x: x.get("ham_prob", 0), reverse=True)
    # Sort by judge score (best spam quality first)
    filtered.sort(key=lambda x: x.get("judge_score", 0), reverse=True)
    
    # Deduplicate by SMS text (keep highest ham_prob version)
    seen_texts = set()
    unique = []
    for sample in filtered:
        text = sample.get("sms", "").strip().lower()
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique.append(sample)
    
    print(f"[ADV] {len(unique)} unique samples after deduplication")
    
    selected = unique[:num_samples]
    print(f"[ADV] Selected {len(selected)} best bypass samples")
    
    # Save selected bypasses to JSON with round-specific filename
    selected_path = Path(bypass_log_path).parent / f"selected_bypasses_round_{round_num}.json"
    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"[ADV] Saved selected bypasses to {selected_path}")
    
    return selected


def add_adversarial_to_dataset(
    selected_samples: list[dict],
    spam_file: str = "data/spam_messages.json",
    train_ratio: float = 0.8,
    seed: int = 42,
    round_num: int = 0,
) -> dict:
    """
    Add adversarial samples to the spam dataset.
    
    Args:
        selected_samples: List of bypass samples to add.
        spam_file: Path to spam_messages.json.
        train_ratio: Fraction for training (rest goes to validation).
        seed: Random seed for train/val split.
        round_num: FL round number (for metadata).
        
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
            "round": round_num,  # Track which round generated this
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
    num_samples_to_add: int = DEFAULT_ADVERSARIAL_SAMPLES,
    total_episodes: int = DEFAULT_ADVERSARIAL_EPISODES,
    batch_size: int = 64,
    min_ham_prob: float = 0.6,
    wandb_run_id: str = None,
    sft_epochs: int = DEFAULT_SFT_EPOCHS,
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
        wandb_run_id=wandb_run_id,
        sft_epochs=sft_epochs,
    )
    
    # Step 2: Select best bypass samples
    selected = select_best_bypasses(
        bypass_log_path=bypass_log_path,
        num_samples=num_samples_to_add,
        min_ham_prob=min_ham_prob,
        round_num=round_num,  # Pass round number
    )
    
    # Step 3: Add to dataset
    dataset_stats = add_adversarial_to_dataset(
        selected_samples=selected,
        train_ratio=0.8,
        round_num=round_num,  # Pass round number
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
