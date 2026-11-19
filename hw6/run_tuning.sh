#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

gpu="${1:?UUID GPU}"
scratch="${2:?Папка с окружением и весами}"
mkdir -p logs failed
for config in baseline prefill1024 interactive ngram; do
  for attempt in 1 2 3; do
    log="logs/tuning-$config-attempt-$attempt-$(date +%s).log"
    if [[ -e "results/tuning/$config/120/profile/profile_export_aiperf.json" ]]; then
      code=0
      break
    fi
    if [[ -e "results/tuning/$config/120" ]]; then
      mv "results/tuning/$config" "failed/$config-$(date +%s)-$$"
    fi
    code=0
    bash run_series.sh "$gpu" "$scratch" "$config" tuning 120 >"$log" 2>&1 || code=$?
    if [[ "$code" -ne 125 && "$code" -ne 137 ]]; then
      break
    fi
    sleep 15
  done
  printf '%s\n' "$code" >"logs/tuning-$config.status"
  printf '%s %s status=%s\n' "$(date -Is)" "$config" "$code"
  if [[ "$config" == baseline && "$code" -ne 0 ]]; then
    exit "$code"
  fi
done
