"""flspam: A Flower / PyTorch app for SMS spam classification."""

import gc
from pathlib import Path

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

# Checkpoint directory
CHECKPOINT_DIR = Path("checkpoints")


# Global references for metric aggregation functions
_wandb_run = None
_current_round = 0
_global_model = None  # Reference to model for checkpointing


def _save_checkpoint(model, round_num: int, metrics: dict = None):
    """Save model checkpoint after aggregation."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"round_{round_num}"
    save_model(model, str(checkpoint_path))
    print(f"[SERVER] Saved checkpoint: {checkpoint_path}")

    # Also save as 'latest' for easy resumption
    latest_path = CHECKPOINT_DIR / "latest"
    save_model(model, str(latest_path))


def train_metrics_aggregator(
    replies: list[RecordDict], weight_key: str
) -> MetricRecord:
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
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for reply in replies:
        metrics = reply.get("metrics", MetricRecord())
        num_examples = metrics.get(weight_key, 1)
        total_examples += num_examples
        weighted_loss += metrics.get("eval_loss", 0.0) * num_examples
        weighted_acc += metrics.get("eval_acc", 0.0) * num_examples
        weighted_precision += metrics.get("eval_precision", 0.0) * num_examples
        weighted_recall += metrics.get("eval_recall", 0.0) * num_examples
        weighted_f1 += metrics.get("eval_f1", 0.0) * num_examples

    avg_loss = weighted_loss / total_examples if total_examples > 0 else 0.0
    avg_acc = weighted_acc / total_examples if total_examples > 0 else 0.0
    avg_precision = weighted_precision / total_examples if total_examples > 0 else 0.0
    avg_recall = weighted_recall / total_examples if total_examples > 0 else 0.0
    avg_f1 = weighted_f1 / total_examples if total_examples > 0 else 0.0

    if _wandb_run is not None:
        wandb.log(
            {
                "eval_loss": avg_loss,
                "eval_acc": avg_acc,
                "eval_precision": avg_precision,
                "eval_recall": avg_recall,
                "eval_f1": avg_f1,
            },
            step=_current_round,
        )

    return MetricRecord(
        {
            "eval_loss": avg_loss,
            "eval_acc": avg_acc,
            "eval_precision": avg_precision,
            "eval_recall": avg_recall,
            "eval_f1": avg_f1,
        }
    )


from .config_defaults import (  # noqa: E402
    DEFAULT_ADVERSARIAL_EPISODES,
    DEFAULT_ADVERSARIAL_SAMPLES,
    DEFAULT_SFT_EPOCHS,
)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global _wandb_run, _current_round, _global_model
    _current_round = 0

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    _spam_strategy: str = context.run_config.get("spam-strategy", "iid")  # noqa: F841
    _spam_alpha: float = context.run_config.get("spam-alpha", 0.5)  # noqa: F841
    save_every: int = context.run_config.get(
        "save-every", 1
    )  # Save checkpoint every N rounds

    # Adversarial training config
    adversarial_rounds_str: str = context.run_config.get("adversarial-rounds", "5")
    # Parse comma-separated string to list of ints
    adversarial_rounds: list = [
        int(r.strip()) for r in adversarial_rounds_str.split(",") if r.strip()
    ]
    adversarial_episodes: int = context.run_config.get(
        "adversarial-episodes", DEFAULT_ADVERSARIAL_EPISODES
    )
    adversarial_samples: int = context.run_config.get(
        "adversarial-samples", DEFAULT_ADVERSARIAL_SAMPLES
    )
    sft_epochs: int = context.run_config.get("sft-epochs", DEFAULT_SFT_EPOCHS)
    grpo_steps: int = context.run_config.get("grpo-steps", 60)
    grpo_num_generations: int = context.run_config.get("grpo-num-generations", 8)
    grpo_batch_size: int = context.run_config.get("grpo-batch-size", 4)

    # Initialize W&B
    # Start with all run config values
    wandb_config = dict(context.run_config)
    # Add computed/parsed values
    wandb_config.update(
        {
            "num_clients": get_num_clients(),
            "model": "ModernBERT-base + LoRA",
            "adversarial_rounds_list": adversarial_rounds,  # Log parsed list alongside string
        }
    )

    _wandb_run = wandb.init(
        project="flspam",
        name=f"fedavg-{num_rounds}r-{get_num_clients()}c",
        config=wandb_config,
    )

    # Load global model with LoRA
    global_model = get_model(use_lora=True)
    _global_model = global_model

    # Only share trainable parameters (LoRA adapters + classifier head)
    trainable_state = {
        k: v
        for k, v in global_model.state_dict().items()
        if "lora" in k.lower() or "classifier" in k.lower()
    }
    arrays = ArrayRecord(trainable_state)

    # Initialize FedAvg strategy with custom metric aggregation for W&B logging
    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_evaluate_nodes=2,
        train_metrics_aggr_fn=train_metrics_aggregator,
        evaluate_metrics_aggr_fn=eval_metrics_aggregator,
    )

    # Run federated learning round by round with checkpointing
    print(f"\n[SERVER] Starting {num_rounds} rounds of federated learning...")
    print(f"[SERVER] Checkpointing every {save_every} round(s) to {CHECKPOINT_DIR}/")

    current_arrays = arrays
    result = None

    for round_num in range(1, num_rounds + 1):
        print(f"\n[SERVER] ===== Round {round_num}/{num_rounds} =====")

        # Run a single round
        result = strategy.start(
            grid=grid,
            initial_arrays=current_arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=1,  # Single round at a time
        )

        # Update arrays for next round
        current_arrays = result.arrays

        # Update global model with aggregated weights
        global_model.load_state_dict(result.arrays.to_torch_state_dict(), strict=False)

        # Save checkpoint
        if round_num % save_every == 0 or round_num == num_rounds:
            eval_metrics = getattr(result, "eval_metrics", None) or {}
            metrics = {
                "loss": eval_metrics.get("eval_loss", 0) if eval_metrics else 0,
                "acc": eval_metrics.get("eval_acc", 0) if eval_metrics else 0,
                "f1": eval_metrics.get("eval_f1", 0) if eval_metrics else 0,
            }
            _save_checkpoint(global_model, round_num, metrics)

        # Run adversarial training at specified rounds
        if round_num in adversarial_rounds:
            print(f"\n[SERVER] 🎯 Running adversarial training at round {round_num}...")

            # Save current model for adversarial training
            adversarial_detector_path = f"checkpoints/round_{round_num}_adversarial"
            save_model(global_model, adversarial_detector_path)

            # Clear FL model from GPU to make room for adversarial models
            global_model.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            try:
                # Import and run adversarial training
                from src.adversarial.fl_integration import run_adversarial_round

                adversarial_stats = run_adversarial_round(
                    detector_path=adversarial_detector_path,
                    round_num=round_num,
                    output_base_dir="adversarial_outputs",
                    num_samples_to_add=adversarial_samples,
                    total_episodes=adversarial_episodes,
                    wandb_run_id=_wandb_run.id if _wandb_run else None,
                    sft_epochs=sft_epochs,
                    grpo_steps=grpo_steps,
                    grpo_num_generations=grpo_num_generations,
                    grpo_batch_size=grpo_batch_size,
                )

                print("[SERVER] ✓ Adversarial training complete!")
                print(
                    f"[SERVER]   Found {adversarial_stats['num_bypasses_found']} bypasses"
                )
                print(
                    f"[SERVER]   Added {adversarial_stats['added_total']} samples to dataset"
                )

            except Exception as e:
                print(f"[SERVER] ⚠ Adversarial training failed: {e}")
                import traceback

                traceback.print_exc()

            finally:
                # Clear adversarial models and reload FL model to GPU
                gc.collect()
                torch.cuda.empty_cache()
                global_model.to("cuda" if torch.cuda.is_available() else "cpu")
                print("[SERVER] Restored FL model to GPU")

    # Log final metrics to W&B
    final_eval_metrics = getattr(result, "eval_metrics", None) or {} if result else {}
    if final_eval_metrics:
        wandb.log(
            {
                "final/loss": final_eval_metrics.get("eval_loss", 0),
                "final/accuracy": final_eval_metrics.get("eval_acc", 0),
                "final/precision": final_eval_metrics.get("eval_precision", 0),
                "final/recall": final_eval_metrics.get("eval_recall", 0),
                "final/f1": final_eval_metrics.get("eval_f1", 0),
            }
        )

    # Save final model to disk
    print("\n[SERVER] Saving final model to disk...")
    save_model(global_model, "final_model")

    # Log model artifact to W&B
    artifact = wandb.Artifact("spam-classifier", type="model")
    artifact.add_dir("final_model")
    wandb.log_artifact(artifact)

    # Also log checkpoints as artifact
    if CHECKPOINT_DIR.exists():
        checkpoint_artifact = wandb.Artifact("checkpoints", type="model-checkpoints")
        checkpoint_artifact.add_dir(str(CHECKPOINT_DIR))
        wandb.log_artifact(checkpoint_artifact)

    wandb.finish()
    _wandb_run = None
    _global_model = None
