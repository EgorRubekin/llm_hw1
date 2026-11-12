#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 data.py
python3 evaluate.py Qwen/Qwen3.5-2B --output results/base
for experiment in exact partial dr_grpo; do
  python3 train.py "configs/$experiment.json"
  python3 evaluate.py "runs/$experiment/merged" --output "results/$experiment"
done
python3 summarize.py
