"""Run all v3 experiments, restartable. Results go to <out>/<experiment>/ in small chunk files, so an interruption
(Ctrl-C, sleep, crash) loses at most a minute of work; running the same command again continues where it stopped.

  python3 run_v3.py run      [--nproc 6] [--out ../data/v3] [--only a,b]   run everything that is not done yet
  python3 run_v3.py status   [--out ../data/v3]                             progress per experiment
  python3 run_v3.py collect  [--out ../data/v3]                             merge chunks -> <exp>.parquet (+ _agents.pkl)
Test options: --limit N (first N jobs per experiment), --tiny (small network, short runs: checks the plumbing only).
"""
import os, sys, time, glob, pickle, argparse, datetime
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def log(out, msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(os.path.join(out, "progress.log"), "a") as fh:
        fh.write(line + "\n")


def tiny(kw):
    kw = dict(kw)
    if isinstance(kw.get("graph"), str):
        kw.pop("graph"); kw.update(N=300, m=3, p_tri=0.5)
    kw["N"] = min(kw.get("N", 300), 300)
    T = int(kw.get("T", 2000)); kw["T"] = max(T // 10, 60); kw["burn"] = kw["T"] // 2
    if kw.get("snap_times") is not None:
        kw["snap_times"] = sorted({min(s // 10, kw["T"]) for s in kw["snap_times"]})
    if kw.get("int_end", 0) > 0:
        kw["int_start"] //= 10; kw["int_end"] = kw["int_start"] + max((kw["int_end"] - kw["int_start"]) // 10, 1)
    return kw


def done_indices(d):
    idx = set()
    for f in glob.glob(os.path.join(d, "chunk_*.parquet")):
        try:
            idx |= set(pd.read_parquet(f, columns=["job_idx"]).job_idx.tolist())
        except Exception:
            pass                                     # a half-written file from a hard kill: its jobs are simply re-run
    return idx


def flush(d, rows, pays):
    if not rows:
        return
    stamp = f"{int(time.time() * 1000)}_{os.getpid()}"
    if pays:
        with open(os.path.join(d, f"payload_{stamp}.pkl.tmp"), "wb") as fh:
            pickle.dump(pays, fh)
        os.replace(os.path.join(d, f"payload_{stamp}.pkl.tmp"), os.path.join(d, f"payload_{stamp}.pkl"))
    pd.DataFrame(rows).to_parquet(os.path.join(d, f"chunk_{stamp}.parquet.tmp"))
    os.replace(os.path.join(d, f"chunk_{stamp}.parquet.tmp"), os.path.join(d, f"chunk_{stamp}.parquet"))


def run_experiment(name, build, out, nproc, limit=None, is_tiny=False, flush_every=60.0):
    import exp_v3 as X
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    jobs = build()
    if limit:
        jobs = jobs[:limit]
    done = done_indices(d)
    todo = [(i, tiny(kw) if is_tiny else kw, meta, kind) for i, (kw, meta, kind) in enumerate(jobs) if i not in done]
    if not todo:
        log(out, f"{name}: all {len(jobs)} runs done"); return True
    log(out, f"{name}: {len(todo)} of {len(jobs)} runs to do")
    rows, pays, t0, last = [], {}, time.time(), time.time()
    n_new = 0
    with Pool(nproc) as pool:
        try:
            for idx, s, pay in pool.imap_unordered(X.run_job, todo, chunksize=1):
                rows.append(s); n_new += 1
                if pay is not None:
                    pays[idx] = (todo_meta(jobs, idx), pay)
                if time.time() - last > flush_every:
                    flush(d, rows, pays); rows, pays = [], {}; last = time.time()
                    rate = n_new / (time.time() - t0); left = len(todo) - n_new
                    log(out, f"{name}: {len(done) + n_new}/{len(jobs)} done, ~{left / max(rate, 1e-9) / 60:.0f} min left for this experiment")
        except KeyboardInterrupt:
            flush(d, rows, pays); log(out, f"{name}: interrupted, progress saved"); raise
    flush(d, rows, pays)
    log(out, f"{name}: finished ({n_new} runs in {(time.time() - t0) / 60:.1f} min)")
    return True


def todo_meta(jobs, idx):
    return jobs[idx][1]


def collect(out):
    import exp_v3 as X
    for name, build in X.EXPERIMENTS:
        d = os.path.join(out, name)
        files = sorted(glob.glob(os.path.join(d, "chunk_*.parquet")))
        if not files:
            continue
        D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        D = D.drop_duplicates("job_idx", keep="last").sort_values("job_idx").reset_index(drop=True)
        D.to_parquet(os.path.join(out, f"{name}.parquet"))
        pays = {}
        for f in sorted(glob.glob(os.path.join(d, "payload_*.pkl"))):
            pays.update(pickle.load(open(f, "rb")))
        if pays:
            pickle.dump([pays[k] for k in sorted(pays)], open(os.path.join(out, f"{name}_agents.pkl"), "wb"))
        n_total = len(build())
        print(f"{name}: {len(D)}/{n_total} runs" + (f", {len(pays)} with agent data" if pays else ""))


def status(out):
    import exp_v3 as X
    tot = 0; dn = 0
    for name, build in X.EXPERIMENTS:
        n = len(build()); k = len(done_indices(os.path.join(out, name))); tot += n; dn += min(k, n)
        print(f"{name:24s} {k:5d} / {n:5d}")
    print(f"{'total':24s} {dn:5d} / {tot:5d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "collect"])
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "v3"))
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("DFP_NPROC", "6")))
    ap.add_argument("--only", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import exp_v3 as X
    if a.cmd == "status":
        status(a.out)
    elif a.cmd == "collect":
        collect(a.out)
    else:
        import engine as E
        E.run(seed=0, T=5, N=50); E.run(seed=0, T=5, N=50, K=2)        # compile once (cached for the workers)
        only = set(a.only.split(",")) if a.only else None
        if not a.tiny:                                                   # build network caches once, before the workers start
            import config as C
            for net in ("twitter", "twitter_rewired", "fb_jhu", "twitch"):
                C.graph(net)
        log(a.out, f"start: {a.nproc} workers, output in {os.path.abspath(a.out)}")
        for name, build in X.EXPERIMENTS:
            if only and name not in only:
                continue
            run_experiment(name, build, a.out, a.nproc, limit=a.limit, is_tiny=a.tiny)
        collect(a.out)
        log(a.out, "all experiments finished; merged results written next to the chunk folders")
