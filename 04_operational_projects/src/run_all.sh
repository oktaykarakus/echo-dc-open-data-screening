#!/bin/bash
# stage 04: run every step with all temp/cache paths on the external drive. Usage: bash run_all.sh [script.py ...]
# Interpreter: python3 (3.9.6; pandas, numpy, scipy, matplotlib, pyproj). No downloads, no venv, no network access by code.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PE="$(dirname "$HERE")"
mkdir -p "$PE/.tmp/tmp" "$PE/.tmp/cache" "$PE/.tmp/mpl" "$PE/.tmp/pyc"
export TMPDIR="$PE/.tmp/tmp" TMP="$PE/.tmp/tmp" TEMP="$PE/.tmp/tmp" XDG_CACHE_HOME="$PE/.tmp/cache" \
       MPLCONFIGDIR="$PE/.tmp/mpl" PYTHONPYCACHEPREFIX="$PE/.tmp/pyc" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
cd "$HERE"
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(e1_evidence.py e2_campus_groups.py e6_manual_review.py e3_plausibility.py e4_campus_sensitivity.py e5_figures.py test_phase_e.py)
for s in "${STEPS[@]}"; do echo "=== $s"; python3 -B "$s"; done
