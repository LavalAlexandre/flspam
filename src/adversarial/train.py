#!/usr/bin/env python3
"""
GRPO Adversarial Spam Generator Training Script.

Train a Qwen3-4B-Instruct-2507 (non-thinking) model to generate spam SMS 
that bypasses a ModernBERT detector.

Usage:
    uv run src/adversarial/train.py --detector-path final_model
    uv run src/adversarial/train.py --detector-path final_model --max-steps 1000
    uv run src/adversarial/train.py --help
"""

import argparse
import os
import sys

# Disable vLLM to avoid expandable_segments conflict on Colab
# Colab pre-initializes PyTorch CUDA with expandable_segments which is incompatible with vLLM
os.environ["UNSLOTH_DISABLE_VLLM"] = "1"

import wandb

from .config import GRPOSpamConfig
from .rewards import RewardFunctions
from .trainer import (
    evaluate_bypass_rate,
    export_adversarial_samples,
    train_adversarial_generator,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Use config defaults as the single source of truth
    defaults = GRPOSpamConfig()
    
    parser = argparse.ArgumentParser(
        description="Train adversarial spam generator using GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--detector-path",
        type=str,
        default=defaults.detector_path,
        help="Path to fine-tuned ModernBERT spam detector",
    )

    # Model settings
    parser.add_argument(
        "--generator-model",
        type=str,
        default=defaults.generator_model,
        help="Generator model name or path",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=defaults.lora_rank,
        help="LoRA rank for generator",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load generator in 4-bit quantization (auto-detected if not set)",
    )

    # Training settings
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=defaults.total_episodes,
        help="Total prompt episodes (steps = episodes / batch_size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Per-device batch size (auto-detected based on GPU if not set)",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=defaults.num_generations,
        help="Number of generations per prompt for GRPO",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=defaults.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=defaults.num_samples,
        help="Number of training samples to generate",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults.output_dir,
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--lora-output-dir",
        type=str,
        default=defaults.lora_output_dir,
        help="Output directory for final LoRA adapter",
    )

    # Evaluation
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation",
    )

    # Export
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip exporting adversarial samples",
    )
    parser.add_argument(
        "--export-samples",
        type=int,
        default=defaults.export_num_samples,
        help="Number of samples to export",
    )
    parser.add_argument(
        "--export-threshold",
        type=float,
        default=defaults.export_bypass_threshold,
        help="Minimum HAM probability for exported samples",
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default=defaults.export_path,
        help="Path to save exported samples",
    )

    # W&B
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=defaults.wandb_project,
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=defaults.wandb_run_name,
        help="W&B run name",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Random seed",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=defaults.print_every,
        help="Print sample every N steps",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Build config from args
    # Pass None for auto-detected values if not explicitly set via CLI
    config = GRPOSpamConfig(
        generator_model=args.generator_model,
        detector_path=args.detector_path,
        lora_rank=args.lora_rank,
        load_in_4bit=args.load_in_4bit if args.load_in_4bit else None,  # Auto-detect if not set
        total_episodes=args.total_episodes,
        per_device_batch_size=args.batch_size if args.batch_size else None,  # Auto-detect if not set
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        lora_output_dir=args.lora_output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        seed=args.seed,
        print_every=args.print_every,
        export_num_samples=args.export_samples,
        export_bypass_threshold=args.export_threshold,
        export_path=args.export_path,
    )

    # Train
    model, tokenizer, trainer = train_adversarial_generator(config)

    # Evaluate
    if not args.skip_eval:
        reward_funcs = RewardFunctions(
            detector_path=config.detector_path,
            print_every=1000,  # Don't print during eval
        )
        evaluate_bypass_rate(
            model,
            tokenizer,
            reward_funcs,
            config.lora_output_dir,
            num_eval=args.eval_samples,
        )

    # Export
    if not args.skip_export:
        reward_funcs = RewardFunctions(
            detector_path=config.detector_path,
            print_every=1000,
        )
        export_adversarial_samples(
            model,
            tokenizer,
            reward_funcs,
            config.lora_output_dir,
            config,
        )

    # Cleanup
    wandb.finish()
    print("\n✓ Training complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
