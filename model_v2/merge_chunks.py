"""Merge the chunk files written by chunk_runner.py into one parquet (dropping duplicate jobs).
Usage: python3 merge_chunks.py out/phase_twitter out/phase_twitter.parquet"""
import sys, glob, os, pandas as pd

if __name__ == "__main__":
    d, outp = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(os.path.join(d, "chunk_*.parquet")))
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("job_idx").sort_values("job_idx")
    D.to_parquet(outp); print(f"{len(files)} chunks -> {len(D)} runs -> {outp}")
