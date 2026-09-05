#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/root/autodl-tmp/AIC-VideoHighlight-run"
DATASET_DIR="${DATASET_DIR:-/root/autodl-tmp/datasets}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
MODEL_REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
COARSE_FPS="${COARSE_FPS:-2.0}"
export HF_HOME="/root/autodl-tmp/hf-cache"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
source /root/miniconda3/bin/activate aic-video-highlight

exec vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --allowed-local-media-path "${DATASET_DIR}" \
  --media-io-kwargs "{\"video\":{\"fps\":${COARSE_FPS}}}"
