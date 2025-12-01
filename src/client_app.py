"""flspam: A Flower / PyTorch app for SMS spam classification."""

import torch
from flwr.client import ClientApp
from flwr.common import Context, Message, ArrayRecord, MetricRecord, RecordDict

from src.task import get_model, load_data
from src.task import test as test_fn
from src.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = get_model(use_lora=True)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with spam distribution strategy from config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    spam_strategy = context.run_config.get("spam-strategy", "iid")
    spam_alpha = context.run_config.get("spam-alpha", 0.5)
    
    trainloader, _ = load_data(
        partition_id, 
        num_partitions,
        spam_strategy=spam_strategy,
        spam_alpha=spam_alpha,
    )

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
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
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = get_model(use_lora=True)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data with spam distribution strategy from config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    spam_strategy = context.run_config.get("spam-strategy", "iid")
    spam_alpha = context.run_config.get("spam-alpha", 0.5)
    
    _, valloader = load_data(
        partition_id,
        num_partitions,
        spam_strategy=spam_strategy,
        spam_alpha=spam_alpha,
    )

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
