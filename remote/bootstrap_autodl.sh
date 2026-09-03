#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/AIC-VideoHighlight"
export HF_HOME="/root/autodl-tmp/hf-cache"

nvidia-smi
python3 --version
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Python 3.12 is required")'
df -h

mkdir -p /root/autodl-tmp/models /root/autodl-tmp/hf-cache /root/autodl-tmp/datasets

if [[ ! -f "${PROJECT_DIR}/pyproject.toml" ]]; then
  echo "Project not found at ${PROJECT_DIR}; clone/pull it there first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

# Intentionally unpinned until the first Qwen3.5-4B AutoDL compatibility run.
python -m pip install vllm

echo "Bootstrap complete. Record python/vLLM versions after smoke tests pass."
