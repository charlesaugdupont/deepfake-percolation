"""Shared figure style for the paper (PNAS sizes: 1 column = 8.7 cm, 2 columns = 17.8 cm).

Colour: categorical slots come from a palette validated for colour-vision deficiency (dataviz validator: adjacent
pairs CVD dE >= 9, normal-vision dE >= 19); every line also gets its own marker and dash pattern, so series stay
distinguishable in greyscale and where lines overlap. Sequential maps are single-hue light -> dark.
"""
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

CM = 1 / 2.54
W1, W2 = 8.7 * CM, 17.8 * CM

# categorical slots (fixed order; validated set), neutral grey for "none / never", ink colours for text
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
BLUE, ORANGE, AQUA, VIOLET, MAGENTA = CAT[:5]
GREY = "#77756f"                                       # dark enough to stay apart from aqua under deuteranopia
INK, INK2, GRID = "#1d1d1b", "#55544f", "#e7e6e2"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DASHES = ["-", "--", ":", "-.", (0, (5, 1.5)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (8, 2))]

# roles used across figures (identity follows the entity, not the figure)
HIST_ALPHA = 0.72                                       # opacity of overlaid belief histograms (all figures)
NEVER, LOCKED, USER = GREY, ORANGE, BLUE              # agent groups: never checked / trapped after trying / still checks
CHK_A, CHK_B, BOTH, ONLY_ONE = BLUE, ORANGE, AQUA, VIOLET
ACTUAL = "#008300"                                     # the checker's actual J (reference line)
GREEN = ACTUAL; GRAY = GREY; PURPLE = VIOLET; RED = "#e34948"

# sequential maps (one hue, light -> dark)
CM_DIP = LinearSegmentedColormap.from_list("dip", ["#fbfaf7", "#f6d7c3", "#ec9a6c", "#d4572a", "#8f2a12", "#4a1406"])
CM_LOCK = LinearSegmentedColormap.from_list("lock", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
CM_GEN = LinearSegmentedColormap.from_list("gen", ["#f7f7f5", "#cfcde6", "#8f86d6", "#5040b8", "#241a6b"])
# roles of the maps: split of trust (Hartigan's dip) violet; share not checking blue; 'would pay, trapped' orange (as the
# orange 'would pay, trapped' bars and the orange 'checked, then stopped' group)
CM_SPLIT, CM_TRAP = CM_GEN, CM_DIP
# how much further genuine items travel than fakes (reach gap): green, as the 'checked' colour
CM_GAP = LinearSegmentedColormap.from_list("gap", ["#f6faf8", "#c3e8d7", "#6cc9a2", "#1baf7a", "#0f7a53", "#073d2a"])


def blues(n):
    """n shades of the blue ramp (light -> dark), for ordered series such as cost levels."""
    return [CM_LOCK(x) for x in np.linspace(0.35, 1.0, n)]


# Runs in which nothing circulates (genuine items reach fewer than 5% of agents). Decided 5 Nov ("split by measure"):
# behaviour and belief measures (not checking, trapped, ever checked, the dip, believed share of fakes) are averaged
# over the runs in which items circulate; reach and welfare over all runs, since a collapse really means that nothing
# spreads and nobody gains. Map cells stay hatched when the mean reach over all runs is below 5%.
MIN_REACH = 0.05
ALL_RUNS = ("reach_true", "reach_fake", "delta", "welfare_sen", "welfare_mean", "payoff_mean")


def circulating(df):
    """The runs (rows) in which items circulate: genuine items reach at least MIN_REACH of the agents."""
    return df[df.reach_true >= MIN_REACH]


def t95(n):
    """Two-sided 95% critical value of Student's t with n - 1 degrees of freedom (n = number of runs; array-friendly,
    NaN where n < 2). A 95% CI of a mean over n runs is mean +- t95(n) * SD / sqrt(n). 1.96 is only the limit for many
    runs: 12.7 for 2 runs, 4.30 for 3, 2.26 for 10, 2.09 for 20 (all bands used 1.96 until 5 Nov)."""
    from scipy import stats
    n = np.asarray(n, float)
    with np.errstate(invalid="ignore"):
        return np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)), np.nan)


def ci95_hw(sd, n):
    """Half-width of the 95% CI (Student's t) of a mean over n runs with standard deviation sd; 0 where n < 2."""
    n = np.asarray(n, float)
    return np.nan_to_num(t95(n) * np.asarray(sd, float) / np.sqrt(np.maximum(n, 1)))


def agg_ci(df, by, col):
    """Per value of `by`: mean of df[col] over runs, its 95% CI half-width (Student's t) and the number of runs n."""
    g = df.groupby(by)[col].agg(["mean", "std", "count"])
    g["hw"] = ci95_hw(g["std"].fillna(0).values, g["count"].values)
    return g.rename(columns={"count": "n"})


def setup():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        # legends inside a plot get a light box (legends placed outside the axes pass frameon=False)
        "legend.frameon": True, "legend.framealpha": 0.92, "legend.facecolor": "white", "legend.edgecolor": "#cfcdc7",
        "legend.fancybox": False, "legend.borderpad": 0.45, "patch.linewidth": 0.5,
        "lines.linewidth": 1.2, "lines.markersize": 3.2, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "dejavusans", "hatch.linewidth": 0.35,
    })


def style(i, color=None):
    """Line style for the i-th series: colour (fixed order), marker and dash pattern."""
    return dict(color=color or CAT[i % len(CAT)], marker=MARKERS[i % len(MARKERS)], ls=DASHES[i % len(DASHES)],
                markersize=3.0, markeredgewidth=0.4, markeredgecolor="white")


def panel_label(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def hexmap(ax, x, y, z, dx, dy, cmap, vmin, vmax, mask=None, hatch_mask=None):
    """Draw a value per lattice cell as a hexagon. x: share of fakes, y: log10 cost (rows dy apart, odd rows shifted
    by dx/2). The hexagons are pointy-topped and tile the plane in axis units (aspect set so they look regular)."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    # pointy-top hexagon of width dx and row spacing dy: circumradius r with vertical scale dy / (1.5 r)
    r = dx / np.sqrt(3)
    sy = dy / (1.5 * r)                                    # vertical stretch so rows land dy apart
    patches, vals, hatched = [], [], []
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        if mask is not None and mask[i]:
            continue
        hp = RegularPolygon((xi, yi / sy), numVertices=6, radius=r * 1.0, orientation=0)
        patches.append(hp); vals.append(zi); hatched.append(bool(hatch_mask[i]) if hatch_mask is not None else False)
    from matplotlib.transforms import Affine2D
    tr = Affine2D().scale(1, sy) + ax.transData
    pc = PatchCollection([p for p, h in zip(patches, hatched) if not h], cmap=cmap, transform=tr, edgecolor="face", linewidth=0.2)
    pc.set_array(np.array([v for v, h in zip(vals, hatched) if not h])); pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    hp = [p for p, h in zip(patches, hatched) if h]
    if hp:
        ax.add_collection(PatchCollection(hp, transform=tr, facecolor="#e4e2dc", edgecolor="#9a978f", hatch="//////", linewidth=0.0))
    ax.set_xlim(x.min() - dx * 0.6, x.max() + dx * 0.6); ax.set_ylim(y.min() - dy * 0.75, y.max() + dy * 0.75)
    return pc


def cvd_simulate(png_in, out_prefix):
    """Write simulated deuteranopia, protanopia and tritanopia versions of a PNG (Machado et al. 2009, severity 1)."""
    from PIL import Image
    M = {"deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
         "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
         "tritan": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]])}
    img = np.asarray(Image.open(png_in).convert("RGB")).astype(float) / 255
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    outs = []
    for k, m in M.items():
        s = np.clip(lin @ m.T, 0, 1)
        srgb = np.where(s <= 0.0031308, 12.92 * s, 1.055 * s ** (1 / 2.4) - 0.055)
        p = f"{out_prefix}_{k}.png"; Image.fromarray((srgb * 255).round().astype(np.uint8)).save(p); outs.append(p)
    return outs
