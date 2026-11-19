#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

url="${URL:-http://localhost:8000}"
data=data/requests.jsonl
output="${1:?Укажите папку результатов}"
rpm="${2:-120}"
count="${3:-$(wc -l < "$data")}"
rps="$(awk -v rpm="$rpm" 'BEGIN { printf "%.12g", rpm / 60 }')"
mkdir -p "$(dirname "$output")"
mkdir "$output"

tokenizer_args=()
if [[ -n ${TOKENIZER_PATH:-} ]]; then
  tokenizer_args=(--tokenizer "$TOKENIZER_PATH")
fi
trap 'curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null' EXIT
curl -fsS -X POST "$url/reset_prefix_cache" >/dev/null

aiperf profile --url "$url" --model Qwen/Qwen3-8B "${tokenizer_args[@]}" \
  --endpoint-type chat --custom-dataset-type raw_payload \
  --input-file "$data" --dataset-sampling-strategy sequential \
  --warmup-request-count 40 --warmup-concurrency 8 \
  --request-count "$count" --request-rate "$rps" --arrival-pattern constant \
  --streaming --request-timeout-seconds 120 \
  --export-outputs-json --output-artifact-dir "$output/profile"
