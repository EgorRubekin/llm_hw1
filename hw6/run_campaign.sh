#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

gpu="${1:?UUID GPU}"
scratch="${2:?Папка с окружением и весами}"
stage="${3:?spec или load}"

run() {
  local code=0
  for attempt in 1 2 3; do
    code=0
    bash run_series.sh "$gpu" "$scratch" "$@" || code=$?
    if [[ "$code" -ne 125 && "$code" -ne 137 ]]; then
      return "$code"
    fi
    sleep 15
  done
  return "$code"
}

case "$stage" in
  spec)
    run plain_v1 tuning 120
    for config in plain_v1 ngram; do
      run "$config" spec 60 120 180 240 360
    done
    ;;
  load)
    config="$(python3 metrics.py best)"
    printf 'Конфигурация для SLA: %s\n' "$config"

    probe() {
      local rpm="$1"
      local result="results/load/$config/$rpm/profile/profile_export_aiperf.json"
      if [[ ! -s "$result" ]]; then
        run "$config" load "$rpm" >&2
      fi
      python3 metrics.py load "$config" "$rpm"
    }

    low=0
    high=0
    rpm=180
    while (( low == 0 || high == 0 )); do
      verdict="$(probe "$rpm")"
      printf '%s RPM %s\n' "$rpm" "$verdict"
      if [[ "$verdict" == pass* ]]; then
        low="$rpm"
        rpm=$((rpm + 20))
      else
        high="$rpm"
        rpm=$((rpm - 20))
      fi
      if (( rpm < 60 || rpm > 1200 )); then
        printf 'Нужна дополнительная точка за пределами диапазона поиска.\n' >&2
        exit 1
      fi
    done

    while (( high - low > 5 )); do
      rpm=$(((high + low) / 10 * 5))
      verdict="$(probe "$rpm")"
      printf '%s RPM %s\n' "$rpm" "$verdict"
      if [[ "$verdict" == pass* ]]; then
        low="$rpm"
      else
        high="$rpm"
      fi
    done
    printf 'SLA выполнен при %s RPM, нарушен при %s RPM\n' "$low" "$high"
    ;;
  *)
    printf 'Неизвестный этап %s\n' "$stage" >&2
    exit 2
    ;;
esac
