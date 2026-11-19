#!/usr/bin/env bash
set -euo pipefail
export VLLM_SERVER_DEV_MODE=1

config=${1:?Укажите конфигурацию}
shift
case "$config" in
  baseline) args=() ;;
  plain_v1) export VLLM_USE_V2_MODEL_RUNNER=0; args=(--no-async-scheduling) ;;
  prefill1024) args=(--max-num-batched-tokens 1024) ;;
  interactive) args=(--performance-mode interactivity) ;;
  ngram) export VLLM_USE_V2_MODEL_RUNNER=0; args=(--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}') ;;
  *) printf 'Неизвестная конфигурация %s\n' "$config" >&2; exit 2 ;;
esac

exec vllm serve Qwen/Qwen3-8B "${args[@]}" "$@"
