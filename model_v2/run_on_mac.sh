#!/bin/bash
# Runs the two long Twitter-network experiments (GSA, then the phase diagram from the end of its job list) on this Mac.
# Safe to stop with Ctrl-C and restart: finished runs are kept in ../data/v2/chunks and skipped next time.
set -u
cd "$(dirname "$0")"
python3 -m pip install --quiet numba numpy scipy pandas pyarrow diptest
export DFP_NETDIR="$(cd ../data/networks && pwd)"
export DFP_NPROC="${DFP_NPROC:-$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 8)}"
echo "using $DFP_NPROC worker processes"
until python3 chunk_runner.py gsa twitter 600 ../data/v2/chunks; do :; done
export DFP_REVERSE=1
until python3 chunk_runner.py phase twitter 600 ../data/v2/chunks; do :; done
echo "ALL DONE"
