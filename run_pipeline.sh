#!/usr/bin/env bash
# =============================================================================
# Full Flickr30K-CFQ pipeline — 4x A100 80GB bare metal
#
# Prerequisites:
#   - conda env 'flickr30k' activated
#   - HuggingFace models downloaded to ~/models/
#   - LLaVA weights at tag/output/LLaVA-13B-v1.1  (or set LLAVA_MODEL below)
#   - Images in ./images/  (or set IMG_DIR below)
#   - Flickr30K test images in ./flickr30k-images/test/
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HOME/models"
IMG_DIR="${REPO_ROOT}/images"
LLAVA_MODEL="${REPO_ROOT}/tag/output/LLaVA-13B-v1.1"
VLLM_PORT=8081
VLLM_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Combine data  (CPU only)
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 1: Combine data ==="
cd "${REPO_ROOT}/combine"
python combineData.py
log "Stage 1 done."

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Image tagging with LLaVA  (uses all 4 GPUs via device_map=auto)
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 2: LLaVA image tagging ==="
cd "${REPO_ROOT}/tag"
python tag.py \
    --img_dir    "${IMG_DIR}" \
    --save_path  data/tag_responses.json \
    --model_path "${LLAVA_MODEL}" \
    --num_gpus   4
python processTagResponses.py
log "Stage 2 done."

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — Query enhancement  (vLLM on GPUs 2,3)
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 3: Start vLLM server ==="
cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL}" \
    --tensor-parallel-size 2 \
    --port ${VLLM_PORT} \
    --max-model-len 4096 \
    --dtype float16 \
    > vllm_server.log 2>&1 &
VLLM_PID=$!
echo "${VLLM_PID}" > vllm_server.pid
log "vLLM PID: ${VLLM_PID} — waiting for it to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        log "vLLM server is up."; break
    fi
    sleep 5
    [[ $i -eq 60 ]] && die "vLLM server failed to start. Check vllm_server.log"
done

log "=== STAGE 3: Enhance queries ==="
cd "${REPO_ROOT}/enhance"
python enhance.py
python deal_raw_enhanced.py
python check.py
log "Stage 3 done."

log "Stopping vLLM server..."
kill "$(cat "${REPO_ROOT}/vllm_server.pid")" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — Encode images  (one model per GPU, all 4 in parallel)
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 4a: Encode images (4 models x 4 GPUs in parallel) ==="
cd "${REPO_ROOT}/encoder"
CUDA_VISIBLE_DEVICES=0 python img_encoder.py --model clip-vit-base-patch32  --model_dir "${MODEL_DIR}" --test_dir "${IMG_DIR}" &
CUDA_VISIBLE_DEVICES=1 python img_encoder.py --model groupvit-gcc-yfcc       --model_dir "${MODEL_DIR}" --test_dir "${IMG_DIR}" &
CUDA_VISIBLE_DEVICES=2 python img_encoder.py --model align-base              --model_dir "${MODEL_DIR}" --test_dir "${IMG_DIR}" &
CUDA_VISIBLE_DEVICES=3 python img_encoder.py --model clipseg-rd64-refined    --model_dir "${MODEL_DIR}" --test_dir "${IMG_DIR}" &
wait
log "Image encoding done."

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4b — Encode raw text  (4 models in parallel)
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 4b: Encode raw text ==="
CUDA_VISIBLE_DEVICES=0 python raw_text_encoder.py --model clip-vit-base-patch32  --model_dir "${MODEL_DIR}" &
CUDA_VISIBLE_DEVICES=1 python raw_text_encoder.py --model groupvit-gcc-yfcc       --model_dir "${MODEL_DIR}" &
CUDA_VISIBLE_DEVICES=2 python raw_text_encoder.py --model align-base              --model_dir "${MODEL_DIR}" &
CUDA_VISIBLE_DEVICES=3 python raw_text_encoder.py --model clipseg-rd64-refined    --model_dir "${MODEL_DIR}" &
wait
log "Raw text encoding done."

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4c — Encode enhanced text  (4 models in parallel)
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 4c: Encode enhanced text ==="
CUDA_VISIBLE_DEVICES=0 python enhanced_text_encoder.py --model clip-vit-base-patch32  --model_dir "${MODEL_DIR}" &
CUDA_VISIBLE_DEVICES=1 python enhanced_text_encoder.py --model groupvit-gcc-yfcc       --model_dir "${MODEL_DIR}" &
CUDA_VISIBLE_DEVICES=2 python enhanced_text_encoder.py --model align-base              --model_dir "${MODEL_DIR}" &
CUDA_VISIBLE_DEVICES=3 python enhanced_text_encoder.py --model clipseg-rd64-refined    --model_dir "${MODEL_DIR}" &
wait
log "Enhanced text encoding done."

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — Build test splits and evaluate
# ─────────────────────────────────────────────────────────────────────────────
log "=== STAGE 5: Build test splits ==="
cd "${REPO_ROOT}/retrieval"
python make_test_data.py

log "=== STAGE 5: Retrieval evaluation ==="
python retrieval.py

log "=== Pipeline complete! ==="
