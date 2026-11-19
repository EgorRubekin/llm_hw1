#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

gpu="${1:?UUID GPU}"
scratch="${2:?Папка с окружением и весами}"
config="${3:?Конфигурация serve.sh}"
phase="${4:?tuning, spec или load}"
shift 4
[[ $# -gt 0 ]]
case "$phase" in tuning|spec|load) ;; *) exit 2;; esac

image="${IMAGE:-vllm/vllm-openai@sha256:14f7be59bea024f60a7cb7076ce5666d7c72751354a3db22cb713af28cffa2cf}"
revision=b968826d9c46dd6066d109eabc6255188de91218
tokenizer=Qwen/Qwen3-8B
port="${PORT:-18739}"
url="http://127.0.0.1:$port"
name="hse-inference-$(id -u)-$$"
root="results/$phase/$config"
mkdir -p "$root" "$scratch"/{vllm,triton,tmp,xcache}
printf '%s\n' '67c48379e5ae2608f7ca08650fd346facdc01f766a3e8eecea6459d7cec00ebc  data/requests.jsonl' | sha256sum -c
used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
[[ "$used" -lt 32 ]]
docker=(docker)
if ! docker info >/dev/null 2>&1; then docker=(sudo -n docker); fi
common=(--rm --pull=never --log-driver=none --network host --ipc host
  --user "$(id -u):$(id -g)" -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" -e HOME=/scratch -e HF_HOME=/scratch/hf -e HF_HUB_OFFLINE=1
  -e TMPDIR=/scratch/tmp -e VLLM_CACHE_ROOT=/scratch/vllm -e TRITON_CACHE_DIR=/scratch/triton
  -e XDG_CACHE_HOME=/scratch/xcache -e OMP_NUM_THREADS=6
  -v "$scratch:/scratch" -v "$PWD:/homework" -w /homework --entrypoint bash)

server=("${docker[@]}" run "${common[@]}" --name "$name" --gpus "device=$gpu"
  -e NVIDIA_REQUIRE_CUDA="cuda>=12.0"
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia/current:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
  "$image" serve.sh "$config" --revision "$revision" --host 127.0.0.1 --port "$port")
stamp="$(date +%Y%m%dT%H%M%S)-$$"
server_log="$root/server-$stamp.log"
printf '%q ' "${server[@]}" >"$root/server-$stamp.command"
printf '\n' >>"$root/server-$stamp.command"
"${server[@]}" >"$server_log" 2>&1 &
server_pid=$!
trap '"${docker[@]}" stop --time 10 "$name" >/dev/null 2>&1 || true; wait "$server_pid" || true' EXIT
trap 'exit 130' INT TERM
for ((i=0; i<180; i++)); do
  if curl -fsS "$url/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    code=0
    wait "$server_pid" || code=$?
    tail -n 60 "$server_log"
    exit "$((code == 0 ? 1 : code))"
  fi
  sleep 5
done
curl -fsS "$url/health" >/dev/null

client=("${docker[@]}" run "${common[@]}" -e "URL=$url" -e "TOKENIZER_PATH=$tokenizer"
  -e PATH=/scratch/benchmark/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "$image")
for rpm; do
  path="$root/$rpm"
  if [[ -s "$path/profile/profile_export_aiperf.json" ]]; then
    continue
  fi
  if [[ -e "$path" ]]; then
    mkdir -p failed
    mv "$path" "failed/$phase-$config-$rpm-$stamp"
  fi
  printf '%s %s %s %s RPM\n' "$(date -Is)" "$phase" "$config" "$rpm"
  curl -fsS "$url/metrics" >"$root/metrics-$rpm-before.txt"
  case "$phase" in
    tuning) "${client[@]}" benchmark.sh "$path" "$rpm" >"$root/client-$rpm.log" 2>&1 ;;
    spec) "${client[@]}" benchmark.sh "$path" "$rpm" 160 >"$root/client-$rpm.log" 2>&1 ;;
    load) "${client[@]}" load_test.sh "$rpm" "$path" >"$root/client-$rpm.log" 2>&1 ;;
  esac
  curl -fsS "$url/metrics" >"$root/metrics-$rpm-after.txt"
  printf '%s %s %s %s RPM завершён\n' "$(date -Is)" "$phase" "$config" "$rpm"
done
