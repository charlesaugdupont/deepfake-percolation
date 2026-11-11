"""SI figure: starting beliefs about the checker (main setting phi = 0.3, l/g = 2): (A) split of trust (Hartigan's dip) over
the shared starting trust J0 and the relative cost of checking, (B) share not checking vs J0 for four prior strengths,
(C, D) spreading the starting beliefs between agents: split of trust (C), shares not checking and ever checked (D). (The starting belief about the share of genuine items mainly sets how
soon checking starts; that comparison is reported in the text, the panels are in v3_figS_baserate.py.)
Usage: v3_figS_starting.py <priors parquet> <out.png>"""
import sys, os, pandas as pd, matplotlib as mpl
mpl.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import v3_fig5_beliefs as checker_beliefs
fs.setup()


def make(priors_parquet, out, hex_parquet=None):
    fig = plt.figure(figsize=(fs.W2, 4.9))
    H = pd.read_parquet(hex_parquet) if hex_parquet and os.path.exists(hex_parquet) else None
    checker_beliefs.draw(fig, fig.add_gridspec(1, 1)[0], pd.read_parquet(priors_parquet), letters="ABCD", H=H)
    fig.savefig(out, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
