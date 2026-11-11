#!/bin/bash
# Runs all v3 experiments on this Mac with 6 workers (change with: DFP_NPROC=5 bash run_on_mac_v3.sh).
# Safe to stop (Ctrl-C) and run again: it continues where it stopped. Results: ../data/v3, progress: ../data/v3/progress.log
set -e
cd "$(dirname "$0")"
VENV="$HOME/.venvs/dfp"
if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up a separate Python environment in $VENV (one time, ~1 minute)..."
  PY=$(command -v python3.13 || command -v python3)
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet numpy pandas pyarrow scipy numba diptest
fi
export DFP_NETDIR="$(cd ../data/networks && pwd)"
export DFP_NPROC="${DFP_NPROC:-6}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p ../data/v3
echo "Running with $DFP_NPROC workers; keep the Mac plugged in with the lid open. Progress: data/v3/progress.log"
caffeinate -i -s "$VENV/bin/python" run_v3.py run --nproc "$DFP_NPROC" --out ../data/v3
