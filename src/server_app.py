"""flspam: A Flower / PyTorch app for SMS spam classification."""

import torch
import wandb
from flwr.common import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.server import ServerApp
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from src.task import get_model, get_num_clients
from src.model.modernbert import save_model

# Create ServerApp
app = ServerApp()


# Global wandb run reference for metric aggregation functions
_wandb_run = None
_current_round = 0


def train_metrics_aggregator(replies: list[RecordDict], weight_key: str) -> MetricRecord:
    """Aggregate training metrics and log to W&B."""
    global _current_round
    
    total_examples = 0
    weighted_loss = 0.0
    
    for reply in replies:
        metrics = reply.get("metrics", MetricRecord())
        num_examples = metrics.get(weight_key, 1)
        train_loss = metrics.get("train_loss", 0.0)
        total_examples += num_examples
        weighted_loss += train_loss * num_examples
    
    avg_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
    
    if _wandb_run is not None:
        wandb.log({"train_loss": avg_loss}, step=_current_round)
    
    return MetricRecord({"train_loss": avg_loss})


def eval_metrics_aggregator(replies: list[RecordDict], weight_key: str) -> MetricRecord:
    """Aggregate evaluation metrics and log to W&B."""
    global _current_round
    _current_round += 1  # Increment after eval (end of round)
    
    total_examples = 0
    weighted_loss = 0.0
    weighted_acc = 0.0
    
    for reply in replies:
        metrics = reply.get("metrics", MetricRecord())
        num_examples = metrics.get(weight_key, 1)
        eval_loss = metrics.get("eval_loss", 0.0)
        eval_acc = metrics.get("eval_acc", 0.0)
        total_examples += num_examples
        weighted_loss += eval_loss * num_examples
        weighted_acc += eval_acc * num_examples
    
    avg_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
    avg_acc = weighted_acc / total_examples if total_examples > 0 else 0.0
    
    if _wandb_run is not None:
        wandb.log({
            "eval_loss": avg_loss,
            "eval_acc": avg_acc,
        }, step=_current_round)
    
    return MetricRecord({"eval_loss": avg_loss, "eval_acc": avg_acc})


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _wandb_run, _current_round
    _current_round = 0

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    spam_strategy: str = context.run_config.get("spam-strategy", "iid")
    spam_alpha: float = context.run_config.get("spam-alpha", 0.5)

    # Initialize W&B
    _wandb_run = wandb.init(
        project="flspam",
        name=f"fedavg-{num_rounds}r-{get_num_clients()}c",
        config={
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "lr": lr,
            "num_clients": get_num_clients(),
            "spam_strategy": spam_strategy,
            "spam_alpha": spam_alpha,
            "model": "ModernBERT-base + LoRA",
        },
    )

    # Load global model with LoRA
    global_model = get_model(use_lora=True)
    
    # Only share trainable parameters (LoRA adapters + classifier head)
    trainable_state = {
        k: v for k, v in global_model.state_dict().items()
        if "lora" in k.lower() or "classifier" in k.lower()
    }
    arrays = ArrayRecord(trainable_state)

    # Initialize FedAvg strategy with custom metric aggregation for W&B logging
    strategy = FedAvg(
        fraction_train=fraction_train,
        train_metrics_aggr_fn=train_metrics_aggregator,
        evaluate_metrics_aggr_fn=eval_metrics_aggregator,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Log final metrics to W&B
    if result.metrics:
        wandb.log({
            "final/loss": result.metrics.get("eval_loss", 0),
            "final/accuracy": result.metrics.get("eval_acc", 0),
        })

    # Save final model to disk
    print("\nSaving final model to disk...")
    # Load result weights into model
    global_model.load_state_dict(result.arrays.to_torch_state_dict(), strict=False)
    save_model(global_model, "final_model")
    
    # Log model artifact to W&B
    artifact = wandb.Artifact("spam-classifier", type="model")
    artifact.add_dir("final_model")
    wandb.log_artifact(artifact)
    
    wandb.finish()
    _wandb_run = None
