#!/usr/bin/env bash
set -euo pipefail

scratch="${1:?Укажите пустую папку для временных файлов и весов}"
image="${IMAGE:-vllm/vllm-openai@sha256:14f7be59bea024f60a7cb7076ce5666d7c72751354a3db22cb713af28cffa2cf}"
revision=b968826d9c46dd6066d109eabc6255188de91218
mkdir -p "$scratch"/{tmp,hf,vllm,triton}
scratch="$(cd "$scratch" && pwd)"
docker=(docker)
if ! docker info >/dev/null 2>&1; then docker=(sudo -n docker); fi
runner=("${docker[@]}" run --rm --pull=never --log-driver=none
  --user "$(id -u):$(id -g)" -e "USER=$(id -un)" -e "LOGNAME=$(id -un)"
  -e HOME=/scratch -e HF_HOME=/scratch/hf -e TMPDIR=/scratch/tmp -e HF_HUB_DISABLE_XET=1
  -v "$scratch:/scratch" --entrypoint bash "$image")
"${runner[@]}" -lc 'python3 -m venv /scratch/benchmark
/scratch/benchmark/bin/pip install --no-cache-dir aiperf==0.13.0 transformers==5.22.1
/scratch/benchmark/bin/aiperf --version' &
client_pid=$!
"${runner[@]}" -lc 'python3 - "$1" <<'"'"'PY'"'"'
import sys
from huggingface_hub import snapshot_download

snapshot_download("Qwen/Qwen3-8B", revision=sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
PY
' prepare "$revision" &
model_pid=$!
wait "$client_pid"
wait "$model_pid"
cache="$scratch/hf/hub/models--Qwen--Qwen3-8B"
mkdir -p "$cache/refs"
printf '%s' "$revision" >"$cache/refs/main"
