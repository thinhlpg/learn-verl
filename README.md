# veRL Shallow Dive

- need `uv` installed
```bash
uv venv
source .venv/bin/activate
pip install -r pyproject.toml
```

We need to install the following pre-requisites: 
https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment

- CUDA: Version >= 12.4
- cuDNN: Version >= 9.8.0
- Apex

## Quick start
https://verl.readthedocs.io/en/latest/start/install.html#install-verl

```bash
# Initialize and update the VERL submodule (pinned to v0.4.1)
git submodule update --init --recursive
cd verl
uv pip install --no-deps -e .
uv pip install flash-attn --no-build-isolation

```
