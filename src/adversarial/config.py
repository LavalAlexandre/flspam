"""Configuration for GRPO adversarial spam training."""

from dataclasses import dataclass, field
from typing import Optional

import torch


def detect_gpu_config() -> dict:
    """Detect GPU type and return optimal configuration.

    Returns:
        Dictionary with GPU-specific settings:
        - gpu_name: Name of the GPU
        - has_bf16: Whether GPU supports BF16
        - has_fp16: Whether GPU supports FP16 efficiently
        - recommended_dtype: Best dtype for this GPU
        - recommended_batch_size: Optimal batch size
        - load_in_4bit: Whether to use 4-bit quantization
        - gpu_memory_gb: GPU memory in GB
    """
    if not torch.cuda.is_available():
        return {
            "gpu_name": "CPU",
            "has_bf16": False,
            "has_fp16": False,
            "recommended_dtype": "float32",
            "recommended_batch_size": 4,
            "load_in_4bit": True,
            "gpu_memory_gb": 0,
        }

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    compute_capability = torch.cuda.get_device_capability(0)
    major, minor = compute_capability

    # BF16 requires compute capability >= 8.0 (Ampere+)
    has_bf16 = major >= 8
    # FP16 with Tensor Cores requires >= 7.0 (Volta+), efficient on >= 7.5 (Turing+)
    has_tensor_cores = major >= 7
    has_efficient_fp16 = major >= 7 and minor >= 5

    # GPU-specific configurations
    # batch = per_device_batch_size (samples per forward pass)
    # For GRPO: effective = batch * num_generations * gradient_accumulation
    # ULTRA AGGRESSIVE settings to maximize GPU utilization!
    # Qwen3-1.7B is small (~3.5GB in bf16), so we can push batch sizes very high
    gpu_configs = {
        # Pascal (P100, GTX 10xx) - No Tensor Cores, FP16 slower than FP32
        "P100": {"dtype": "float32", "batch": 32, "4bit": False, "num_gen": 4},
        "Tesla P": {"dtype": "float32", "batch": 32, "4bit": False, "num_gen": 4},
        "GTX 10": {"dtype": "float32", "batch": 16, "4bit": True, "num_gen": 4},
        # Turing (T4, RTX 20xx) - Tensor Cores, good FP16 (16GB)
        "T4": {"dtype": "float16", "batch": 16, "4bit": False, "num_gen": 8},
        "Tesla T": {"dtype": "float16", "batch": 16, "4bit": False, "num_gen": 8},
        "RTX 20": {"dtype": "float16", "batch": 16, "4bit": False, "num_gen": 8},
        # Ampere (A100, A10, RTX 30xx) - BF16 support
        # A100-80GB: Model ~3.5GB, leaves ~75GB for activations/KV cache
        "A100-SXM4-80GB": {
            "dtype": "bfloat16",
            "batch": 256,
            "4bit": False,
            "num_gen": 16,
        },  # 80GB - ULTRA
        "A100-SXM4-40GB": {
            "dtype": "bfloat16",
            "batch": 128,
            "4bit": False,
            "num_gen": 16,
        },  # 40GB
        "A100-PCIE-80GB": {
            "dtype": "bfloat16",
            "batch": 256,
            "4bit": False,
            "num_gen": 16,
        },  # 80GB PCIe
        "A100-PCIE-40GB": {
            "dtype": "bfloat16",
            "batch": 128,
            "4bit": False,
            "num_gen": 16,
        },  # 40GB PCIe
        "A100": {
            "dtype": "bfloat16",
            "batch": 128,
            "4bit": False,
            "num_gen": 16,
        },  # Default A100
        "A10G": {"dtype": "bfloat16", "batch": 48, "4bit": False, "num_gen": 8},  # 24GB
        "A10": {"dtype": "bfloat16", "batch": 48, "4bit": False, "num_gen": 8},
        "RTX 30": {"dtype": "bfloat16", "batch": 24, "4bit": False, "num_gen": 8},
        "RTX A": {"dtype": "bfloat16", "batch": 24, "4bit": False, "num_gen": 8},
        # Ada Lovelace (L4, RTX 40xx)
        "L4": {"dtype": "bfloat16", "batch": 48, "4bit": False, "num_gen": 8},  # 24GB
        "RTX 40": {"dtype": "bfloat16", "batch": 48, "4bit": False, "num_gen": 8},
        # Hopper (H100)
        "H100": {
            "dtype": "bfloat16",
            "batch": 256,
            "4bit": False,
            "num_gen": 16,
        },  # 80GB - ULTRA
    }

    # Find matching config
    config = None
    for key, cfg in gpu_configs.items():
        if key in gpu_name:
            config = cfg
            break

    # Fallback based on compute capability
    if config is None:
        if has_bf16:
            config = {"dtype": "bfloat16", "batch": 48, "4bit": False, "num_gen": 16}
        elif has_efficient_fp16:
            config = {"dtype": "float16", "batch": 32, "4bit": False, "num_gen": 8}
        elif has_tensor_cores:
            config = {"dtype": "float16", "batch": 24, "4bit": True, "num_gen": 8}
        else:
            config = {"dtype": "float32", "batch": 16, "4bit": True, "num_gen": 4}

    # Adjust batch size based on memory - AGGRESSIVE settings
    # Only reduce for very low VRAM, otherwise trust GPU-specific config
    grad_accum = 1
    num_gen = config.get("num_gen", 8)
    if gpu_memory_gb < 12:
        config["batch"] = min(config["batch"], 8)
        config["4bit"] = True
        grad_accum = 2
        num_gen = min(num_gen, 4)
    elif gpu_memory_gb < 16:
        config["batch"] = min(config["batch"], 16)
        grad_accum = 1
        num_gen = min(num_gen, 8)
    elif gpu_memory_gb < 24:
        config["batch"] = min(config["batch"], 48)
        grad_accum = 1
        num_gen = min(num_gen, 8)
    elif gpu_memory_gb < 48:
        config["batch"] = min(config["batch"], 128)
        grad_accum = 1
        num_gen = min(num_gen, 16)
    else:
        # 48GB+ (A100-40GB, A100-80GB, H100) - use GPU-specific config as-is (256 for 80GB)
        grad_accum = 1

    config["grad_accum"] = grad_accum
    config["num_gen"] = num_gen

    return {
        "gpu_name": gpu_name,
        "has_bf16": has_bf16,
        "has_fp16": has_efficient_fp16,
        "recommended_dtype": config["dtype"],
        "recommended_batch_size": config["batch"],
        "recommended_grad_accum": config["grad_accum"],
        "recommended_num_generations": config["num_gen"],
        "load_in_4bit": config["4bit"],
        "gpu_memory_gb": round(gpu_memory_gb, 1),
        "compute_capability": f"{major}.{minor}",
    }


from ..config_defaults import DEFAULT_ADVERSARIAL_EPISODES  # noqa: E402


@dataclass
class GRPOSpamConfig:
    """Configuration for GRPO spam generator training."""

    # Model settings
    # Qwen3-1.7B with /nothink to disable thinking mode
    generator_model: str = "unsloth/Qwen3-1.7B"
    detector_path: str = "final_model"
    max_seq_length: int = 512
    lora_rank: int = 32
    load_in_4bit: Optional[bool] = None  # Auto-detect if None
    gpu_memory_utilization: float = 0.90  # Use more VRAM
    dtype: Optional[str] = (
        None  # Auto-detect if None ("float32", "float16", "bfloat16")
    )

    # Training settings
    learning_rate: float = 5e-5  # Higher LR for faster convergence in short RL runs
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03  # Minimal warmup for short runs (1 step for 30 steps)
    per_device_batch_size: Optional[int] = None  # Auto-detect if None (used for SFT)
    gradient_accumulation_steps: Optional[int] = None  # Auto-detect if None
    num_generations: Optional[int] = None  # Auto-detect if None
    total_episodes: int = (
        DEFAULT_ADVERSARIAL_EPISODES  # Total prompt episodes (aligned with FL config)
    )
    grpo_steps: int = 60  # Number of GRPO training steps per episode
    grpo_batch_size: int = 4  # Batch size for GRPO (separate from SFT batch size)
    save_steps: int = 100

    # GPU config (populated by auto_configure)
    _gpu_config: dict = field(default_factory=dict)

    def __post_init__(self):
        """Auto-configure GPU-specific settings after initialization."""
        self.auto_configure()

    def auto_configure(self):
        """Auto-detect GPU and configure optimal settings."""
        self._gpu_config = detect_gpu_config()

        # Apply auto-detected settings if not explicitly set
        if self.dtype is None:
            self.dtype = self._gpu_config["recommended_dtype"]

        if self.load_in_4bit is None:
            self.load_in_4bit = self._gpu_config["load_in_4bit"]

        if self.per_device_batch_size is None:
            self.per_device_batch_size = self._gpu_config["recommended_batch_size"]

        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = self._gpu_config.get(
                "recommended_grad_accum", 1
            )

        if self.num_generations is None:
            self.num_generations = self._gpu_config.get(
                "recommended_num_generations", 8
            )

    @property
    def max_steps(self) -> int:
        """Return grpo_steps directly (explicit control over training duration)."""
        return self.grpo_steps

    @property
    def torch_dtype(self):
        """Get torch dtype object."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)

    # Generation settings (tuned to prevent mode collapse)
    temperature: float = 1.0  # Higher temp for more diversity
    top_p: float = 0.95
    top_k: int = 50
    min_p: float = 0.05  # Lower min_p to allow more token variety
    repetition_penalty: float = 1.15  # Discourage repeating tokens/phrases
    max_completion_length: int = 100
    max_prompt_length: int = 256

    # Anti-mode-collapse settings
    diversity_weight: float = 1.0  # Penalty for similar outputs in batch (0 to disable)

    # Dataset settings
    num_samples: int = 5000
    seed: int = 42

    # Output settings
    output_dir: str = "grpo_spam_outputs"
    lora_output_dir: str = "grpo_spam_lora"

    # Logging
    wandb_project: str = "flspam-adversarial"
    wandb_run_name: str = "grpo-spam-generator"
    print_every: int = 1  # Log every step for short runs
    bypass_log_path: str = "bypass_samples.json"

    # Export settings
    export_num_samples: int = 500
    export_bypass_threshold: float = 0.6
    export_path: str = "adversarial_spam_samples.json"

    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            "generator_model": self.generator_model,
            "detector_path": self.detector_path,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate,
            "num_generations": self.num_generations,
            "total_episodes": self.total_episodes,
            "max_steps": self.max_steps,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "diversity_weight": self.diversity_weight,
            "dtype": self.dtype,
            "load_in_4bit": self.load_in_4bit,
            "gpu_name": self._gpu_config.get("gpu_name", "unknown"),
            "gpu_memory_gb": self._gpu_config.get("gpu_memory_gb", 0),
            "compute_capability": self._gpu_config.get("compute_capability", "unknown"),
        }

    def print_gpu_info(self):
        """Print detected GPU configuration."""
        effective_batch = self.per_device_batch_size * self.gradient_accumulation_steps

        print(f"{'=' * 50}")
        print(f"GPU: {self._gpu_config.get('gpu_name', 'unknown')}")
        print(f"  Memory: {self._gpu_config.get('gpu_memory_gb', 0):.1f} GB")
        print(
            f"  Compute Capability: {self._gpu_config.get('compute_capability', 'unknown')}"
        )
        print(f"  BF16 Support: {self._gpu_config.get('has_bf16', False)}")
        print(f"  FP16 Tensor Cores: {self._gpu_config.get('has_fp16', False)}")
        print("Configuration:")
        print(f"  dtype: {self.dtype}")
        print(f"  batch_size: {self.per_device_batch_size}")
        print(f"  gradient_accumulation: {self.gradient_accumulation_steps}")
        print(f"  effective_batch: {effective_batch}")
        print(f"  num_generations: {self.num_generations}")
        print(f"  load_in_4bit: {self.load_in_4bit}")
        print("Training:")
        print(f"  total_episodes: {self.total_episodes}")
        print(f"  max_steps: {self.max_steps}")
        print(f"  samples_per_step: {effective_batch * self.num_generations}")
        print(f"{'=' * 50}")
