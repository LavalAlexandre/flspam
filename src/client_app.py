"""flspam: A Flower / PyTorch app for SMS spam classification."""

import logging
import time

import torch
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
    log.info(f"[P{partition_id}] Model created in {time.time() - t0:.2f}s")
    
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
    
    trainloader, _ = load_data(
        partition_id, 
        num_partitions,
        spam_strategy=spam_strategy,
        spam_alpha=spam_alpha,
    )
    log.info(f"[P{partition_id}] Data loaded in {time.time() - t0:.2f}s - {len(trainloader.dataset)} samples, {len(trainloader)} batches")

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
    log.info(f"[P{partition_id}] Training completed in {time.time() - t0:.2f}s - loss: {train_loss:.4f}")

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
    
    _, valloader = load_data(
        partition_id,
        num_partitions,
        spam_strategy=spam_strategy,
        spam_alpha=spam_alpha,
    )
    log.info(f"[P{partition_id}] Eval data loaded in {time.time() - t0:.2f}s - {len(valloader.dataset)} samples")

    # Call the evaluation function
    log.info(f"[P{partition_id}] Running evaluation...")
    t0 = time.time()
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )
    log.info(f"[P{partition_id}] Eval completed in {time.time() - t0:.2f}s - loss: {eval_loss:.4f}, acc: {eval_acc:.4f}")

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    log.info(f"[P{partition_id}] === EVAL END === Total: {time.time() - start_time:.2f}s")
    return Message(content=content, reply_to=msg)
