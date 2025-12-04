"""flspam: A Flower / PyTorch app for SMS spam classification."""

import json
import logging
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import get_scheduler
from src.model.modernbert import create_model, get_tokenizer, MODEL_NAME, LABEL_TO_ID

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
HAM_FILE = DATA_DIR / "ham_messages.json"
SPAM_FILE = DATA_DIR / "spam_messages.json"


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class SMSDataset(Dataset):
    """Dataset for SMS spam classification."""
    
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

# Cache for data
_tokenizer = None
_ham_data = None
_spam_data = None
_persona_ids = None  # Ordered list of persona UUIDs


def get_cached_tokenizer():
    """Get cached tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = get_tokenizer()
    return _tokenizer


def _load_raw_data():
    """Load and cache raw ham/spam data."""
    global _ham_data, _spam_data, _persona_ids
    
    if _ham_data is None:
        with open(HAM_FILE, 'r', encoding='utf-8') as f:
            _ham_data = json.load(f)
        
        # Get ordered list of unique persona IDs
        _persona_ids = list(dict.fromkeys(m['main_uuid'] for m in _ham_data))
    
    if _spam_data is None:
        with open(SPAM_FILE, 'r', encoding='utf-8') as f:
            _spam_data = json.load(f)
    
    return _ham_data, _spam_data, _persona_ids


def distribute_spam(
    spam_messages: list[dict],
    persona_ids: list[str],
    strategy: Literal["iid", "dirichlet"] = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Distribute spam messages across personas.
    
    Strategy:
    1. Separate adversarial spam (source="adversarial") from regular spam
    2. Distribute regular spam according to strategy (iid/dirichlet)
    3. Add ALL adversarial spam to ALL clients (uniform)
    
    This preserves existing spam allocations while ensuring all clients
    learn from new adversarial examples.
    
    Args:
        spam_messages: List of spam message dicts
        persona_ids: List of persona UUIDs
        strategy: Distribution strategy for regular spam
            - "iid": Equal distribution across all personas
            - "dirichlet": Non-IID using Dirichlet distribution
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping persona_id -> list of spam messages
    """
    rng = np.random.default_rng(seed)
    n_personas = len(persona_ids)
    
    # Separate adversarial spam from regular spam
    adversarial_spam = [msg for msg in spam_messages if msg.get("source") == "adversarial"]
    regular_spam = [msg for msg in spam_messages if msg.get("source") != "adversarial"]
    
    n_regular = len(regular_spam)
    
    # Distribute regular spam according to strategy
    if strategy == "iid":
        # Shuffle and distribute evenly
        indices = rng.permutation(n_regular)
        splits = np.array_split(indices, n_personas)
        
        result = {
            pid: [regular_spam[i] for i in split]
            for pid, split in zip(persona_ids, splits)
        }
    
    elif strategy == "dirichlet":
        # Non-IID: some personas get more spam than others
        proportions = rng.dirichlet([alpha] * n_personas)
        counts = (proportions * n_regular).astype(int)
        
        # Fix rounding errors
        diff = n_regular - counts.sum()
        counts[rng.choice(n_personas, size=abs(diff))] += np.sign(diff)
        
        indices = rng.permutation(n_regular)
        result = {}
        start = 0
        for pid, count in zip(persona_ids, counts):
            result[pid] = [regular_spam[i] for i in indices[start:start + count]]
            start += count
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Add ALL adversarial spam to ALL clients (uniform distribution)
    for pid in persona_ids:
        result[pid].extend(adversarial_spam)
    
    return result


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = 32,
    spam_strategy: Literal["iid", "dirichlet"] = "iid",
    spam_alpha: float = 0.5,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Load data for a specific partition (persona/client).
    
    Args:
        partition_id: The partition (client) ID (0 to num_partitions-1)
        num_partitions: Total number of partitions (must match number of personas)
        batch_size: Batch size for dataloaders
        spam_strategy: How to distribute spam ("iid" or "dirichlet")
        spam_alpha: Dirichlet alpha for non-IID spam distribution
        test_size: Fraction of data for testing
        seed: Random seed
        
    Returns:
        trainloader, testloader
    """
    ham_data, spam_data, persona_ids = _load_raw_data()
    
    # Validate partition count matches personas
    if num_partitions != len(persona_ids):
        raise ValueError(
            f"num_partitions ({num_partitions}) must match number of personas ({len(persona_ids)})"
        )
    
    # Get this client's persona
    persona_id = persona_ids[partition_id]
    
    # Get ham messages for this persona
    persona_ham = [m for m in ham_data if m['main_uuid'] == persona_id]
    
    # Distribute spam and get this persona's share
    spam_distribution = distribute_spam(
        spam_data, persona_ids, strategy=spam_strategy, alpha=spam_alpha, seed=seed
    )
    persona_spam = spam_distribution[persona_id]
    
    # Separate adversarial spam (has pre-assigned train/val split) from regular data
    adversarial_spam = [m for m in persona_spam if m.get("source") == "adversarial"]
    regular_spam = [m for m in persona_spam if m.get("source") != "adversarial"]
    
    # For adversarial spam, respect the pre-assigned split (same across all clients)
    adversarial_train = [m for m in adversarial_spam if m.get("split") == "train"]
    adversarial_val = [m for m in adversarial_spam if m.get("split") == "val"]
    
    # Combine regular ham + regular spam for per-client splitting
    regular_messages = persona_ham + regular_spam
    regular_texts = [m['text'] for m in regular_messages]
    regular_labels = [m['label'] for m in regular_messages]
    
    # Shuffle and split regular data (per-client randomization)
    rng = np.random.default_rng(seed + partition_id)
    indices = rng.permutation(len(regular_texts))
    regular_texts = [regular_texts[i] for i in indices]
    regular_labels = [regular_labels[i] for i in indices]
    
    # Train/test split for regular data
    split_idx = int(len(regular_texts) * (1 - test_size))
    regular_train_texts = regular_texts[:split_idx]
    regular_train_labels = regular_labels[:split_idx]
    regular_test_texts = regular_texts[split_idx:]
    regular_test_labels = regular_labels[split_idx:]
    
    # Add adversarial spam (with pre-assigned splits)
    train_texts = regular_train_texts + [m['text'] for m in adversarial_train]
    train_labels = regular_train_labels + [m['label'] for m in adversarial_train]
    test_texts = regular_test_texts + [m['text'] for m in adversarial_val]
    test_labels = regular_test_labels + [m['label'] for m in adversarial_val]
    
    tokenizer = get_cached_tokenizer()
    
    train_dataset = SMSDataset(train_texts, train_labels, tokenizer)
    test_dataset = SMSDataset(test_texts, test_labels, tokenizer)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


def get_num_clients() -> int:
    """Get the number of available clients (personas)."""
    _, _, persona_ids = _load_raw_data()
    return len(persona_ids)


def compute_class_weights(trainloader: DataLoader) -> list[float]:
    """
    Compute inverse frequency class weights from a dataloader.
    
    Returns weights [w_ham, w_spam] where higher weight = more penalty for errors.
    Uses inverse frequency: w_i = n_total / (n_classes * n_i)
    """
    label_counts = {0: 0, 1: 0}
    
    for batch in trainloader:
        labels = batch["labels"].tolist()
        for label in labels:
            label_counts[label] += 1
    
    total = sum(label_counts.values())
    n_classes = 2
    
    # Inverse frequency weighting
    weights = [
        total / (n_classes * label_counts[0]) if label_counts[0] > 0 else 1.0,  # HAM weight
        total / (n_classes * label_counts[1]) if label_counts[1] > 0 else 1.0,  # SPAM weight
    ]
    
    return weights


# ------------------------------------------------------------------
# Training & Evaluation
# ------------------------------------------------------------------

def train(
    model, 
    trainloader, 
    epochs: int, 
    lr: float, 
    device: torch.device,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
    compile_model: bool = False,
    class_weights: list[float] | None = None,
):
    """Train the model on the training set with optimizations.
    
    Args:
        model: ModernBERT model with LoRA
        trainloader: Training data loader
        epochs: Number of local epochs
        lr: Learning rate
        device: Device to train on
        warmup_ratio: Fraction of steps for linear warmup (default 10%)
        max_grad_norm: Maximum gradient norm for clipping (default 1.0)
        use_amp: Use automatic mixed precision (default True)
        compile_model: Use torch.compile for optimization (default False, can be slow first run)
        class_weights: Optional weights for [HAM, SPAM] classes. Higher weight = more penalty for errors.
        
    Returns:
        Average training loss
    """
    model.to(device)
    model.train()
    
    # Optionally compile the model (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    
    # Setup weighted loss function if class weights provided
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        log.info(f"Using class weights: HAM={class_weights[0]:.2f}, SPAM={class_weights[1]:.2f}")
    else:
        criterion = None  # Use model's built-in loss
    
    # Only optimize trainable parameters (LoRA + classifier)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    # Calculate total steps and warmup steps
    total_steps = len(trainloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    # Learning rate scheduler: linear warmup + cosine decay
    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Mixed precision scaler (only for CUDA)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    
    total_loss = 0.0
    num_batches = 0
    total_batches = len(trainloader) * epochs
    log_interval = max(1, total_batches // 5)  # Log ~5 times during training
    
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=None if criterion else labels,  # Don't pass labels if using custom loss
                )
                
                # Use weighted loss if provided, otherwise use model's built-in loss
                if criterion:
                    loss = criterion(outputs.logits, labels)
                else:
                    # Need to compute loss with labels
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first for proper clipping)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_grad_norm
            )
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # LR scheduler step
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress logging
            if num_batches % log_interval == 0 or num_batches == total_batches:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                log.info(f"  Batch {num_batches}/{total_batches} (epoch {epoch+1}/{epochs}) - loss: {avg_loss:.4f}, lr: {lr:.2e}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def test(model, testloader, device: torch.device, use_amp: bool = True):
    """Evaluate the model on the test set.
    
    Args:
        model: ModernBERT model
        testloader: Test data loader
        device: Device to evaluate on
        use_amp: Use automatic mixed precision (default True)
        
    Returns:
        dict with keys: loss, accuracy, precision, recall, f1
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    model.to(device)
    model.eval()
    
    use_amp = use_amp and device.type == "cuda"
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in testloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            total_loss += outputs.loss.item()
            
            # Predictions: argmax over 2 classes
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    
    # Calculate metrics using sklearn
    # pos_label=1 means spam is the positive class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="binary", pos_label=1, zero_division=0
    )
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels) if all_labels else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ------------------------------------------------------------------
# Model Creation (for FL clients)
# ------------------------------------------------------------------

def get_model(use_lora: bool = True):
    """Create a new model instance for FL training."""
    return create_model(use_lora=use_lora)
