"""Adversarial spam generation using GRPO."""

from .config import GRPOSpamConfig
from .rewards import RewardFunctions
from .trainer import train_adversarial_generator
from .fl_integration import run_adversarial_round, add_adversarial_to_dataset
from .sft import train_sft, generate_from_sft
from .rl_simple import train_rl_phase, DetectorOnlyReward

__all__ = [
    "GRPOSpamConfig",
    "RewardFunctions",
    "train_adversarial_generator",
    "run_adversarial_round",
    "add_adversarial_to_dataset",
    # Two-phase training
    "train_sft",
    "generate_from_sft",
    "train_rl_phase",
    "DetectorOnlyReward",
]
