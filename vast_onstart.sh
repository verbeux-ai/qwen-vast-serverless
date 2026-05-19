#!/bin/bash
set -e

# --- PyWorker env vars ---
export WORKER_PORT="${WORKER_PORT:-3000}"
export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"
export MODEL_LOG="/var/log/vllm.log"
export PYWORKER_REPO="${PYWORKER_REPO:-https://github.com/ImPedro29/qwen-vast-serverless}"
export MODEL_NAME="${MODEL_NAME:-qwen3.6-27b}"
export HF_REPO="${HF_REPO:-AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP}"

# --- Baixar modelo (idempotente) ---
pip install hf-transfer huggingface_hub -q --upgrade
if [ ! -f "/workspace/model/config.json" ]; then
    echo "Baixando modelo $HF_REPO..."
    HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', local_dir='/workspace/model')
"
else
    echo "Modelo já em cache."
fi

# --- vLLM em background ---
mkdir -p /var/log
echo "Iniciando vLLM em background -> $MODEL_LOG"
nohup vllm serve /workspace/model \
    --quantization modelopt \
    --trust-remote-code \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.94 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}' \
    --host 0.0.0.0 --port 8000 \
    --served-model-name "$MODEL_NAME" \
    > "$MODEL_LOG" 2>&1 &

# --- PyWorker (foreground) ---
echo "Baixando start_server.sh do vast-ai/pyworker..."
wget -qO /tmp/start_server.sh https://raw.githubusercontent.com/vast-ai/pyworker/main/start_server.sh
chmod +x /tmp/start_server.sh
exec /tmp/start_server.sh
