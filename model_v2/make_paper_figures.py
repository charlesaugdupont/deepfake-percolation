"""Render all paper and SI figures (PDF + PNG) from the experiment outputs in out/ (or $DFP_OUT).
Usage: python3 make_paper_figures.py OUTDIR [figure names ...]"""
import sys, os, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "figures", "paper_v2")
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DFP_OUT", os.path.join(HERE, "out"))      # experiment outputs (e.g. DFP_OUT=../data/v2)
o = lambda *p: os.path.join(DATA, *p)
CS, CLO, CHI = "0.04329", "0.01233", "0.15199"
FIGS = [
    ("fig1_model_mechanism", ["fig_mechanism.py", o("mech_twitter.pkl"), None]),
    ("fig2_split_over_time", ["fig_time.py", o(f"time_twitter_c{CS}.pkl"), o(f"time_twitter_c{CLO}.pkl"), o(f"time_twitter_c{CHI}.pkl"), None]),
    ("fig3_regimes", ["fig_phase.py", o("phase_twitter.parquet"), None, "twitter"]),
    ("fig4_two_checkers", ["fig_two.py", o("two_twitter"), None, CS]),
    ("fig5_starting_beliefs_and_checkers", ["fig_priors.py", o("priors_twitter.parquet"), o("checkers_twitter"), None]),
    ("fig6_consequences", ["fig_consequences.py", o("phase_twitter.parquet"), None]),
    ("figS1_gsa_pawn", ["fig_gsa.py", o("gsa_twitter.parquet"), None]),
    ("figS2_other_networks", ["fig_networks.py", None]),
    ("figS3_robustness", ["fig_robust.py", o("robust_twitter.parquet"), None, CS]),
    ("figS4_two_checkers_holme_kim", ["fig_two.py", o("two_hk"), None, "0.05"]),
    ("figS5_validation_vs_julia", ["fig_validation.py", o("validation_points.csv"), None]),
]
only = set(sys.argv[2:])
for name, args in FIGS:
    if only and name not in only:
        continue
    for ext in ["pdf", "png"]:
        a = [x if x is not None else os.path.join(OUT, f"{name}.{ext}") for x in args]
        print(">>", name, ext, flush=True)
        subprocess.run([sys.executable, *a], cwd=HERE, check=True, env=dict(os.environ, DFP_OUT=DATA))
