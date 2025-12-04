from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ModernBERT for SMS Spam Classification
# https://huggingface.co/answerdotai/ModernBERT-base

MODEL_NAME = "answerdotai/ModernBERT-base"

# Default LoRA configuration for ModernBERT
DEFAULT_LORA_CONFIG = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,  # LoRA rank
    lora_alpha=16,  # LoRA alpha (scaling factor)
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=["Wqkv", "Wo"],  # ModernBERT attention layers
    bias="none",  # Don't train bias terms
)


def get_tokenizer():
    """Get the tokenizer for ModernBERT."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def create_model(
    use_lora: bool = True,
    lora_config: LoraConfig | None = None,
):
    """Create a ModernBERT classifier for binary spam detection.

    Args:
        use_lora: Whether to apply LoRA adapters (default True for fine-tuning)
        lora_config: Custom LoRA config, uses DEFAULT_LORA_CONFIG if None

    Returns:
        Model ready for training with Trainer or manual loop
    """
    # Load ModernBERT with classification head (2 classes for binary)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,  # Binary classification with CrossEntropyLoss
    )

    # Apply LoRA if requested
    if use_lora:
        config = lora_config or DEFAULT_LORA_CONFIG
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    return model


def save_model(model, path: str):
    """Save model - saves LoRA adapters if using PEFT."""
    model.save_pretrained(path)


def load_model(path: str, use_lora: bool = True):
    """Load a saved model."""
    if use_lora:
        # Load base model first, then LoRA adapters
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=1,
            problem_type="single_label_classification",
        )
        return PeftModel.from_pretrained(base_model, path)
    else:
        return AutoModelForSequenceClassification.from_pretrained(path)


# Label mapping for spam classification
LABEL_MAP = {
    0: "ham",  # Normal message
    1: "spam",  # Spam message
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
