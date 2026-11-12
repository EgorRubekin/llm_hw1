#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/report"

pandoc --from=markdown+link_attributes-implicit_figures --pdf-engine=xelatex \
  -V mainfont=Arial -V papersize=a4 -V geometry:margin=19mm \
  -V fontsize=10pt -V colorlinks=true -V linkcolor=blue \
  -o report.pdf report.md
