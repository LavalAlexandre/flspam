# FLspam: Federated Learning for SMS Spam Detection

A Flower / PyTorch application for federated SMS spam classification using **ModernBERT + LoRA**, with synthetic data generation using LLM-powered persona-based conversations.

## Project Structure

```
FLspam/
├── src/
│   ├── client_app.py          # Flower client for federated training
│   ├── server_app.py          # Flower server with W&B logging
│   ├── task.py                # Data loading & training logic
│   ├── model/
│   │   └── modernbert.py      # ModernBERT + LoRA classifier
│   └── synthetic_data/
│       ├── config.py          # Configuration constants
│       ├── text_utils.py      # SMS text cleaning & validation
│       ├── generator.py       # Core SMS generation logic
│       ├── personas.py        # Persona network generator
│       ├── sms_generation.py  # Backward-compatible entry point
│       └── sms_colab_generation.ipynb  # Colab notebook for generation
├── tests/
│   ├── test_dataset.py        # Dataset processing tests
│   └── test_flower_config.py  # Flower configuration & import tests
├── data/
│   ├── personas.json          # Generated personas (by UUID)
│   ├── relationships.json     # Relationship graph between personas
│   └── conversations.json     # Generated SMS conversations
└── pyproject.toml
```

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .

# For GPU generation (Linux with CUDA)
pip install "distilabel[vllm]"

# Login to W&B for experiment tracking
wandb login
```

## Quick Start

### 1. Generate Synthetic Data (or use existing)

```bash
# Generate persona network
uv run python -m src.synthetic_data.personas

# Generate SMS conversations (requires GPU)
python -m src.synthetic_data.sms_generation
```

### 2. Run Federated Learning

```bash
# Run federated training
flwr run .
```

## Model Architecture

**ModernBERT-base + LoRA** for efficient federated fine-tuning:

- Base: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (~150M params)
- LoRA: rank=8, alpha=16, targeting attention layers (`Wqkv`, `Wo`)
- Output: Single logit with sigmoid (binary classification)
- **Trainable params**: ~300K (0.2% of total)

Only LoRA adapters + classifier head are exchanged during FL, reducing communication cost.

## Spam Distribution Strategies

Configure in `pyproject.toml`:

```toml
[tool.flwr.app.config]
spam-strategy = "iid"      # or "dirichlet"
spam-alpha = 0.5           # Dirichlet concentration (lower = more non-IID)
```

| Strategy | Description |
|----------|-------------|
| `iid` | Equal spam distribution across all clients |
| `dirichlet` | Non-IID: some clients get more spam (α controls skewness) |

## Synthetic Data Generation

### Generate Persona Network

Creates main personas with ~100+ contacts each using [nvidia/Nemotron-Personas-USA](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA).

```bash
uv run python -m src.synthetic_data.personas
```

**Features:**
- Smart matching: partners by marital status, colleagues by occupation, friends by shared hobbies
- Configurable contact distribution (family, friends, colleagues, professionals, etc.)
- Multiple conversations per relationship
- Outputs: `data/personas.json`, `data/relationships.json`

### Generate SMS Conversations

Generates realistic SMS conversations using **distilabel + vLLM with Qwen3**.

```bash
# On Linux with NVIDIA GPU (T4, A10, etc.)
python -m src.synthetic_data.generator

# Or use the Colab notebook: src/synthetic_data/sms_colab_generation.ipynb
```

**Features:**
- Batched generation with vLLM + prefix caching for efficiency
- SMS style adapted to persona age/education
- Scenario-based conversations
- Quality controls (length, repetition, cleanup)

**Module Structure:**
- `config.py` - All configuration (paths, model, vLLM settings, scenarios)
- `text_utils.py` - Message cleaning, quality checks, conversation end detection
- `generator.py` - Core generation with `generate_conversations()`, `build_prompt()`
- `sms_generation.py` - Re-exports everything for backward compatibility

## Experiment Tracking

W&B integration logs:
- Per-round metrics (train_loss, eval_loss, eval_acc)
- Final model as artifact
- Run configuration

View experiments at: https://wandb.ai/your-username/flspam

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=src
```

## Configuration

Edit `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 5
fraction-train = 0.8
local-epochs = 2
lr = 1e-4
spam-strategy = "iid"
spam-alpha = 0.5

[tool.flwr.federations.local-simulation]
options.num-supernodes = 20  # Must match persona count
```

## Resources

- [Flower Documentation](https://flower.ai/docs/)
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- [PEFT/LoRA](https://huggingface.co/docs/peft)
- [Nemotron-Personas-USA Dataset](https://huggingface.co/datasets/nvidia/Nemotron-Personas-USA)
- [W&B Documentation](https://docs.wandb.ai/)

## Citation

```bibtex
@article{salman2024investigating,
  title={Investigating Evasive Techniques in SMS Spam Filtering: A Comparative Analysis of Machine Learning Models},
  author={Salman, Muhammad and Ikram, Muhammad and Kaafar, Mohamed Ali},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```

- Paper: https://ieeexplore.ieee.org/document/10431737
- Dataset: https://github.com/smspamresearch/spstudy
