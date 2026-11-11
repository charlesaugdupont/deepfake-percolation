"""Resumable, time-boxed runner: runs the not-yet-done jobs of an experiment in parallel until a time budget is used up,
then writes the finished runs to a new chunk file. Call it repeatedly (e.g. from short remote shell sessions).
Usage: python3 chunk_runner.py <experiment> <net> <budget_seconds> <out_dir>
Exit code 0 = nothing left to do, 2 = more to do. Env: DFP_NETDIR (networks folder), DFP_NPROC (workers),
DFP_REVERSE=1 (take jobs from the end of the list, so two machines can share one job list).
Experiments: phase (all three loss/gain ratios), gsa."""
import sys, os, time, glob, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import engine as E
from sweep import summarize
from gsa import lhs


def phase_jobs(net, reps=6, reps_other=3):
    G = C.grid(net); jobs = []
    for lam in [1.0, 2.0, 4.0]:
        for phi in G["phis"]:
            for c in G["costs"]:
                for r in range(reps if lam == 2.0 else reps_other):
                    kw = C.kwargs(net, phi=float(phi), cost=float(c), rm=lam, seed=int(1e6 * lam + 1e4 * r + 7))
                    jobs.append((kw, dict(net=net, lam=lam, phi=float(phi), cost=float(c), rep=r)))
    return jobs


def gsa_jobs(net, n=1000):
    from exp_gsa import BOUNDS
    X = lhs(n, BOUNDS, np.random.default_rng(2026)); jobs = []
    for i, r in X.iterrows():
        t0, f0, w = 0.5 + r.J0 / 2, 0.5 - r.J0 / 2, r.spread
        kw = C.kwargs(net, phi=r.phi, cost=r.cost, rm=r.lam, tpr=r.tpr, fpr=r.fpr, seed=int(60000 + i), prior_mode=3,
                      t_lo=max(t0 - w, 0.01), t_hi=min(t0 + w, 0.99), f_lo=max(f0 - w, 0.01), f_hi=min(f0 + w, 0.99), kappa=r.kappa)
        jobs.append((kw, dict(i=i, **r.to_dict())))
    return jobs


def phase2_jobs(net, reps=None):
    """Robustness networks: loss/gain = 2 only."""
    G = C.grid(net); jobs = []
    reps = reps or (3 if net.startswith("twitter") else 4)
    for phi in G["phis"]:
        for c in G["costs"]:
            for r in range(reps):
                kw = C.kwargs(net, phi=float(phi), cost=float(c), rm=2.0, seed=int(2e6 + 1e4 * r + 7))
                jobs.append((kw, dict(net=net, lam=2.0, phi=float(phi), cost=float(c), rep=r)))
    return jobs


def robust_jobs(net, T=8000, reps=3, costs=(0.01233, 0.04329)):
    from exp_robust import VARIANTS, SNAPS
    jobs = []
    for vname, v in VARIANTS.items():
        for c in costs:
            for r in range(reps):
                kw = C.kwargs(net, phi=0.3, cost=c, seed=500 + r, T=T, burn=T // 2, snap_times=[s for s in SNAPS if s <= T])
                kw.update(v)
                jobs.append((kw, dict(variant=vname, net=net, cost=c, rep=r)))
    return jobs


BUILDERS = {"phase": phase_jobs, "gsa": gsa_jobs, "phase2": phase2_jobs, "robust": robust_jobs}
OPTS = {"robust": dict(snap_metrics=True)}


def _one(args):
    idx, kw, extra, opts = args
    R = E.run(**kw); s = summarize(R, **opts); s.update(extra); s["job_idx"] = idx
    return s


def done_indices(d):
    idx = set()
    for f in glob.glob(os.path.join(d, "chunk_*.parquet")):
        idx |= set(pd.read_parquet(f, columns=["job_idx"]).job_idx.tolist())
    return idx


if __name__ == "__main__":
    exp, net, budget, out_dir = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    first = int(sys.argv[5]) if len(sys.argv) > 5 else 0          # optional: only take jobs with index >= first ...
    stride = int(sys.argv[6]) if len(sys.argv) > 6 else 1          # ... and index % stride == first % stride (split work)
    t_start = time.time()
    d = os.path.join(out_dir, f"{exp}_{net}"); os.makedirs(d, exist_ok=True)
    jobs = BUILDERS[exp](net)
    done = done_indices(d)
    todo = [(i, kw, ex, OPTS.get(exp, {})) for i, (kw, ex) in enumerate(jobs) if i not in done and i >= first and (i - first) % stride == 0]
    if os.environ.get("DFP_REVERSE") == "1":
        todo = todo[::-1]                                          # work from the end (to share a job list between machines)
    print(f"{exp}/{net}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to do in this share", flush=True)
    if not todo:
        print("NOTHING LEFT", flush=True); sys.exit(0)
    C.graph(net); E.run(seed=0, T=5, N=50)
    res = []
    import multiprocessing as mp
    p = Pool(C.NPROC)
    try:
        it = p.imap_unordered(_one, todo, chunksize=1)
        while len(res) < len(todo):
            left = budget - (time.time() - t_start)
            if left <= 0:
                break
            try:
                res.append(it.next(timeout=left))       # hard deadline: never wait past the budget
            except mp.TimeoutError:
                break
            except StopIteration:
                break
    finally:
        p.terminate()
    if res:
        pd.DataFrame(res).to_parquet(os.path.join(d, f"chunk_{int(time.time())}_{os.getpid()}.parquet"))
    print(f"finished {len(res)} runs in {time.time() - t_start:.0f}s; total done {len(done) + len(res)}/{len(jobs)}", flush=True)
    sys.exit(2 if len(res) < len(todo) else 0)              # 2 = more to do (so `until ...; do :; done` keeps going)
