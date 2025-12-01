"""flspam: A Flower / PyTorch app for SMS spam classification."""

import logging
import os
import time

import torch
import wandb
from flwr.client import ClientApp
from flwr.common import Context, Message, ArrayRecord, MetricRecord, RecordDict

from src.task import get_model, load_data
from src.task import test as test_fn
from src.task import train as train_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [CLIENT] %(message)s")
log = logging.getLogger(__name__)

# Flower ClientApp
app = ClientApp()

def _log_to_wandb(partition_id: int, context: Context, metrics: dict) -> None:
    """Log metrics to wandb, handling run lifecycle properly for Ray actors."""
    try:
        # Use wandb.log with commit=True to avoid issues with run state
        # Each call is a separate "step" - wandb will auto-increment
        run_name = f"client-{partition_id}"
        spam_strategy = context.run_config.get("spam-strategy", "iid")
        spam_alpha = context.run_config.get("spam-alpha", 0.5)
        
        # Always init a new run (reinit=True handles cleanup)
        # Use resume="allow" to continue if run exists
        run = wandb.init(
            project="flspam",
            name=run_name,
            id=f"client-{partition_id}-{context.run_config.get('run-id', 'default')}",
            group="federated-clients",
            job_type="client",
            config={
                "partition_id": partition_id,
                "num_partitions": context.node_config["num-partitions"],
                "spam_strategy": spam_strategy,
                "spam_alpha": spam_alpha,
                "local_epochs": context.run_config.get("local-epochs", 2),
                "batch_size": context.run_config.get("batch-size", 32),
            },
            reinit=True,
            resume="allow",
        )
        
        wandb.log(metrics)
        wandb.finish()
        
    except Exception as e:
        log.warning(f"[P{partition_id}] Failed to log to wandb: {e}")


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""
    partition_id = context.node_config["partition-id"]
    start_time = time.time()
    
    log.info(f"[P{partition_id}] === TRAIN START ===")
    
    # Load the model and initialize it with the received weights
    log.info(f"[P{partition_id}] Loading model...")
    t0 = time.time()
    model = get_model(use_lora=True)
    log.info(f"[P{partition_id}] Model created in {time.time() - t0:.2f}s"))
    
    log.info(f"[P{partition_id}] Loading state dict from server...")
    t0 = time.time()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    log.info(f"[P{partition_id}] State dict loaded in {time.time() - t0:.2f}s")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"[P{partition_id}] Using device: {device}")
    model.to(device)

    # Load the data with spam distribution strategy from config
    log.info(f"[P{partition_id}] Loading data...")
    t0 = time.time()
    num_partitions = context.node_config["num-partitions"]
    spam_strategy = context.run_config.get("spam-strategy", "iid")
    spam_alpha = context.run_config.get("spam-alpha", 0.5)
    batch_size = context.run_config.get("batch-size", 32)
    
    trainloader, _ = load_data(
        partition_id, 
        num_partitions,
        batch_size=batch_size,
        spam_strategy=spam_strategy,
        spam_alpha=spam_alpha,
    )
    log.info(f"[P{partition_id}] Data loaded in {time.time() - t0:.2f}s - {len(trainloader.dataset)} samples, {len(trainloader)} batches (bs={batch_size})")

    # Call the training function
    log.info(f"[P{partition_id}] Starting training: {context.run_config['local-epochs']} epochs, lr={msg.content['config']['lr']}")
    t0 = time.time()
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )
    training_time = time.time() - t0
    log.info(f"[P{partition_id}] Training completed in {training_time:.2f}s - loss: {train_loss:.4f}")
    
    # Log to wandb
    _log_to_wandb(partition_id, context, {
        "train_loss": train_loss,
        "train_time_s": training_time,
        "num_samples": len(trainloader.dataset),
    })

    # Construct and return reply Message
    log.info(f"[P{partition_id}] Preparing response...")
    # Only send trainable parameters (LoRA adapters + classifier)
    trainable_state = {
        k: v for k, v in model.state_dict().items() 
        if any(p.requires_grad for p in [model.get_parameter(k)] if p is not None)
    }
    model_record = ArrayRecord(trainable_state)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    
    log.info(f"[P{partition_id}] === TRAIN END === Total: {time.time() - start_time:.2f}s")
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""
    partition_id = context.node_config["partition-id"]
    start_time = time.time()
    
    log.info(f"[P{partition_id}] === EVAL START ===")

    # Load the model and initialize it with the received weights
    log.info(f"[P{partition_id}] Loading model for eval...")
    model = get_model(use_lora=True)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"[P{partition_id}] Using device: {device}")
    model.to(device)

    # Load the data with spam distribution strategy from config
    log.info(f"[P{partition_id}] Loading eval data...")
    t0 = time.time()
    num_partitions = context.node_config["num-partitions"]
    spam_strategy = context.run_config.get("spam-strategy", "iid")
    spam_alpha = context.run_config.get("spam-alpha", 0.5)
    batch_size = context.run_config.get("batch-size", 32)
    
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size=batch_size,
        spam_strategy=spam_strategy,
        spam_alpha=spam_alpha,
    )
    log.info(f"[P{partition_id}] Eval data loaded in {time.time() - t0:.2f}s - {len(valloader.dataset)} samples")

    # Call the evaluation function
    log.info(f"[P{partition_id}] Running evaluation...")
    t0 = time.time()
    eval_metrics = test_fn(
        model,
        valloader,
        device,
    )
    eval_time = time.time() - t0
    log.info(
        f"[P{partition_id}] Eval completed in {eval_time:.2f}s - "
        f"loss: {eval_metrics['loss']:.4f}, acc: {eval_metrics['accuracy']:.4f}, "
        f"precision: {eval_metrics['precision']:.4f}, recall: {eval_metrics['recall']:.4f}, "
        f"f1: {eval_metrics['f1']:.4f}"
    )
    
    # Log to wandb
    _log_to_wandb(partition_id, context, {
        "eval_loss": eval_metrics["loss"],
        "eval_acc": eval_metrics["accuracy"],
        "eval_precision": eval_metrics["precision"],
        "eval_recall": eval_metrics["recall"],
        "eval_f1": eval_metrics["f1"],
        "eval_time_s": eval_time,
        "num_eval_samples": len(valloader.dataset),
    })

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_metrics["loss"],
        "eval_acc": eval_metrics["accuracy"],
        "eval_precision": eval_metrics["precision"],
        "eval_recall": eval_metrics["recall"],
        "eval_f1": eval_metrics["f1"],
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    log.info(f"[P{partition_id}] === EVAL END === Total: {time.time() - start_time:.2f}s")
    return Message(content=content, reply_to=msg)
