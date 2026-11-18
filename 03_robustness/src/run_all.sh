#!/bin/bash
# stage 03: run every step with all temp/cache paths on the external drive. Usage: bash run_all.sh [script.py ...]
# (default: full pipeline). Interpreter: python3 (3.9.6; pandas, numpy, scipy, matplotlib). No downloads, no venv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(dirname "$HERE")"
mkdir -p "$PD/.tmp/tmp" "$PD/.tmp/cache" "$PD/.tmp/mpl" "$PD/.tmp/pyc"
export TMPDIR="$PD/.tmp/tmp" TMP="$PD/.tmp/tmp" TEMP="$PD/.tmp/tmp" XDG_CACHE_HOME="$PD/.tmp/cache" \
       MPLCONFIGDIR="$PD/.tmp/mpl" PYTHONPYCACHEPREFIX="$PD/.tmp/pyc" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
cd "$HERE"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(d0_verify.py d1_composite_vs_simple.py d2_population.py d3_ablation.py d4_normalisation.py d5_weighting.py d6_crossscale.py d7_figures.py test_phase_d.py)
for s in "${STEPS[@]}"; do echo "=== $s"; python3 -B "$s"; done
