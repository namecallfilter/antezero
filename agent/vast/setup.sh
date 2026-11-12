#!/usr/bin/env bash
# One-time setup of a fresh Vast.ai instance (Ubuntu image, as root, from the unpacked repo):
# build tools, Rust, uv, the Python env (builds the Rust extension), the vastai CLI, then a CUDA check.
set -euo pipefail
cd "$(dirname "$0")/../.."
export DEBIAN_FRONTEND=noninteractive PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
apt-get update -qq && apt-get install -y -qq build-essential curl ca-certificates > /dev/null
command -v cargo > /dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --locked
uv tool install -q vastai
uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('ready:', torch.__version__, torch.cuda.get_device_name(0))"
