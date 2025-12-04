#!/usr/bin/env python3
"""
Two-Phase Adversarial Spam Generator Training.

Phase 1: SFT on spam dataset (learn spam style)
Phase 2: RL with detector-only reward (learn to evade)

Usage:
    # Run both phases
    uv run -m src.adversarial.train_twophase --detector-path final_model --spam-path data/spam_messages.json
    
    # Run only SFT
    uv run -m src.adversarial.train_twophase --detector-path final_model --sft-only
    
    # Run only RL (requires existing SFT adapter)
    uv run -m src.adversarial.train_twophase --detector-path final_model --rl-only --sft-adapter sft_spam_lora
"""

import argparse
import os
import sys

# Disable vLLM
os.environ["UNSLOTH_DISABLE_VLLM"] = "1"

import wandb

from .config import GRPOSpamConfig
from .sft import train_sft, generate_from_sft
from .rl_simple import train_rl_phase

# Create default config to get default values (single source of truth)
_DEFAULT_CONFIG = GRPOSpamConfig()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Defaults are pulled from GRPOSpamConfig to avoid duplication.
    """
    parser = argparse.ArgumentParser(
        description="Two-phase adversarial spam training: SFT → RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--detector-path",
        type=str,
        default=_DEFAULT_CONFIG.detector_path,
        help="Path to fine-tuned ModernBERT spam detector",
    )
    parser.add_argument(
        "--spam-path",
        type=str,
        default="data/spam_messages.json",
        help="Path to spam messages JSON for SFT",
    )

    # Phase selection
    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="Run only SFT phase",
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Run only RL phase (requires --sft-adapter)",
    )
    parser.add_argument(
        "--sft-adapter",
        type=str,
        default="sft_spam_lora",
        help="Path to SFT adapter (for RL phase)",
    )

    # Model settings (from config)
    parser.add_argument(
        "--generator-model",
        type=str,
        default=_DEFAULT_CONFIG.generator_model,
        help="Generator model name (non-thinking variant)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=_DEFAULT_CONFIG.lora_rank,
        help="LoRA rank",
    )

    # SFT settings
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=3,
        help="Number of SFT epochs",
    )
    parser.add_argument(
        "--sft-output",
        type=str,
        default="sft_spam_lora",
        help="Output directory for SFT adapter",
    )

    # RL settings (from config)
    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=_DEFAULT_CONFIG.total_episodes,
        help="Total RL episodes",
    )
    parser.add_argument(
        "--rl-output",
        type=str,
        default="rl_spam_lora",
        help="Output directory for RL adapter",
    )
    parser.add_argument(
        "--use-judge",
        action="store_true",
        help="Use LLM judge for deceptiveness scoring",
    )
    parser.add_argument(
        "--judge-weight",
        type=float,
        default=0.6,
        help="Weight of LLM judge score (0-1)",
    )

    # Common settings (from config)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,  # Auto-detect from GPU
        help="Batch size (auto-detected if not set)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_DEFAULT_CONFIG.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_CONFIG.seed,
        help="Random seed",
    )

    # W&B (from config)
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=_DEFAULT_CONFIG.wandb_project,
        help="W&B project name",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Build config
    config = GRPOSpamConfig(
        generator_model=args.generator_model,
        detector_path=args.detector_path,
        lora_rank=args.lora_rank,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        wandb_project=args.wandb_project,
    )

    print("=" * 60)
    print("Two-Phase Adversarial Spam Training")
    print("=" * 60)
    print(f"Phase 1 (SFT): {'SKIP' if args.rl_only else 'RUN'}")
    print(f"Phase 2 (RL):  {'SKIP' if args.sft_only else 'RUN'}")
    print("=" * 60)

    # Phase 1: SFT
    if not args.rl_only:
        print("\n" + "=" * 60)
        print("PHASE 1: Supervised Fine-Tuning on Spam")
        print("=" * 60)
        
        wandb.init(
            project=args.wandb_project,
            name="sft-spam-generator",
            config={"phase": "sft", **config.to_dict()},
        )
        
        model, tokenizer, _ = train_sft(
            config=config,
            spam_path=args.spam_path,
            output_dir=args.sft_output,
            num_epochs=args.sft_epochs,
        )
        
        # Generate some samples to verify
        print("\n--- Sample generations from SFT model ---")
        samples = generate_from_sft(model, tokenizer, num_samples=3)
        for i, sample in enumerate(samples):
            print(f"{i+1}. {sample[:100]}...")
        
        wandb.finish()
        
        # Update adapter path for RL phase
        sft_adapter = args.sft_output
    else:
        sft_adapter = args.sft_adapter

    # Phase 2: RL
    if not args.sft_only:
        reward_type = "Combined (Detector + LLM Judge)" if args.use_judge else "Detector-Only"
        print("\n" + "=" * 60)
        print(f"PHASE 2: RL with {reward_type} Reward")
        print("=" * 60)
        
        train_rl_phase(
            config=config,
            sft_adapter_path=sft_adapter,
            output_dir=args.rl_output,
            total_episodes=args.rl_episodes,
            use_judge=args.use_judge,
            judge_weight=args.judge_weight,
        )

    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)
    if not args.rl_only:
        print(f"SFT adapter: {args.sft_output}/")
    if not args.sft_only:
        print(f"RL adapter:  {args.rl_output}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
