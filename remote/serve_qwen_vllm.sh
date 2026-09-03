#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/AIC-VideoHighlight"
DATASET_DIR="${DATASET_DIR:-/root/autodl-tmp/datasets}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
COARSE_FPS="${COARSE_FPS:-2.0}"
export HF_HOME="/root/autodl-tmp/hf-cache"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
source .venv/bin/activate

# TODO: verify these flags against the first working AutoDL vLLM version, then pin it.
exec vllm serve "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --allowed-local-media-path "${DATASET_DIR}" \
  --media-io-kwargs "{\"video\":{\"fps\":${COARSE_FPS}}}"
