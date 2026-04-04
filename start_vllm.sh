#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Starts a vLLM OpenAI-compatible server on GPUs 2 and 3 (leaving 0,1 free
# for encoding if you want to run both stages simultaneously).
#
# Edit MODEL to whichever model you have downloaded / have HF access to:
#   - meta-llama/Meta-Llama-3-8B-Instruct   (fast, good quality)
#   - meta-llama/Meta-Llama-3-70B-Instruct  (best quality, needs all 4 GPUs)
#   - lmsys/vicuna-13b-v1.5                 (original paper model)
#
# For Llama-3 models you need to accept the license on HuggingFace and run:
#   huggingface-cli login
# ─────────────────────────────────────────────────────────────────────────────

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
TP_SIZE=2          # tensor parallel — number of GPUs for the LLM
GPU_IDS="2,3"      # which GPUs to use (keeps 0,1 free for encoding)
PORT=8081
LOG_FILE="vllm_server.log"

echo "Starting vLLM server on GPUs ${GPU_IDS} with TP=${TP_SIZE}"
echo "Model: ${MODEL}"
echo "Port:  ${PORT}"
echo "Logs:  ${LOG_FILE}"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size ${TP_SIZE} \
    --port ${PORT} \
    --max-model-len 4096 \
    --dtype float16 \
    > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"
echo "${VLLM_PID}" > vllm_server.pid

# Wait for it to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server is up!"
        break
    fi
    sleep 5
    echo "  still waiting... (${i}/60)"
done
