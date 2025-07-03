# veRL Shallow Dive

This project document my self learning of RL training with veRL library on the GSM8K mathematical reasoning dataset.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) - A fast Python package and project manager
- Python >= 3.11
- CUDA >= 12.4 and cuDNN >= 9.8.0

## Installation

1. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate
```

2. Install the project dependencies:
```bash
pip install -e .
```

3. Install additional pre-requisites following the [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment):
   - CUDA: Version >= 12.4
   - cuDNN: Version >= 9.8.0
   - Apex

## Quick start

Follow the [veRL installation guide](https://verl.readthedocs.io/en/latest/start/install.html#install-verl) for complete setup:

```bash
# Initialize and update the VERL submodule (pinned at commit 8d9e350e, tag v0.4.1)
git submodule update --init --recursive
cd verl
uv pip install --no-deps -e .
uv pip install flash-attn --no-build-isolation
```

## Training

Run the PPO training script:
```bash
./train_ppo_gsm8k.sh
```

The script will train a Qwen2.5-0.5B model on the GSM8K dataset with optimized memory settings for single GPU training.
