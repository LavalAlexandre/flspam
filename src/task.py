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
    
    Args:
        spam_messages: List of spam message dicts
        persona_ids: List of persona UUIDs
        strategy: Distribution strategy
            - "iid": Equal distribution across all personas
            - "dirichlet": Non-IID using Dirichlet distribution
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping persona_id -> list of spam messages
    """
    rng = np.random.default_rng(seed)
    n_personas = len(persona_ids)
    n_spam = len(spam_messages)
    
    if strategy == "iid":
        # Shuffle and distribute evenly
        indices = rng.permutation(n_spam)
        splits = np.array_split(indices, n_personas)
        
        return {
            pid: [spam_messages[i] for i in split]
            for pid, split in zip(persona_ids, splits)
        }
    
    elif strategy == "dirichlet":
        # Non-IID: some personas get more spam than others
        proportions = rng.dirichlet([alpha] * n_personas)
        counts = (proportions * n_spam).astype(int)
        
        # Fix rounding errors
        diff = n_spam - counts.sum()
        counts[rng.choice(n_personas, size=abs(diff))] += np.sign(diff)
        
        indices = rng.permutation(n_spam)
        result = {}
        start = 0
        for pid, count in zip(persona_ids, counts):
            result[pid] = [spam_messages[i] for i in indices[start:start + count]]
            start += count
        
        return result
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


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
    
    # Combine ham and spam
    all_messages = persona_ham + persona_spam
    texts = [m['text'] for m in all_messages]
    labels = [m['label'] for m in all_messages]
    
    # Shuffle
    rng = np.random.default_rng(seed + partition_id)
    indices = rng.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    # Train/test split
    split_idx = int(len(texts) * (1 - test_size))
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
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
        
    Returns:
        Average training loss
    """
    model.to(device)
    model.train()
    
    # Optionally compile the model (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model, mode="reduce-overhead")
    
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
