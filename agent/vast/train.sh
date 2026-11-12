#!/usr/bin/env bash
# Runs az-train on a Vast.ai instance, then tries to stop the instance so the GPU stops billing.
# shortcut: Vast's per-instance key may read and start but not stop instances (401, tested
# 2026-10-19), so the stop can fail; then a monitor on another machine must stop it. Upgrade
# by running `vastai set api-key` on the instance with a key allowed to stop it.
# Start it detached:  nohup bash agent/vast/train.sh --preset base --name l40s_1 > train.out 2>&1 &
# To restart training without stopping the instance, kill this script before az-train.
set -u
cd "$(dirname "$0")/../.."
# Late-game states are longer, so batch shapes vary; this avoids fragmentation OOMs.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# SSH sessions don't inherit Vast's variables; the instance's onstart command
# (`env | grep _ >> /etc/environment`, as Vast's docs advise) saves them there.
[ -n "${CONTAINER_ID:-}" ] || export $(grep -E '^CONTAINER_(ID|API_KEY)=' /etc/environment)
uv run az-train "$@"
echo "[vast] az-train exited with $? at $(date -u); stopping instance ${CONTAINER_ID:-?}"
vastai stop instance "${CONTAINER_ID:-}" --api-key "${CONTAINER_API_KEY:-}"
