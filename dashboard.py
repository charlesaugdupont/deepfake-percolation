import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import time
from dataclasses import fields

from config import SimConfig
from simulation import Simulation


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(layout="wide")


# ============================================================
# SESSION STATE INIT
# ============================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.sim = None
    st.session_state.cascade_id = 0
    st.session_state.running = False

    st.session_state.verify_series = []
    st.session_state.true_series = []
    st.session_state.fake_series = []
    st.session_state.welfare_series = []

    st.session_state.true_total = 0.0
    st.session_state.fake_total = 0.0
    st.session_state.true_count = 0
    st.session_state.fake_count = 0
    st.session_state.welfare_total = 0.0


# ============================================================
# SIDEBAR — AUTO PARAM CONTROLS
# ============================================================

st.sidebar.title("Parameters")

def build_config():
    kwargs = {}
    default_cfg = SimConfig()

    for f in fields(SimConfig):
        value = getattr(default_cfg, f.name)

        if isinstance(value, bool):
            kwargs[f.name] = st.sidebar.checkbox(f.name, value=value)

        elif isinstance(value, int):
            kwargs[f.name] = st.sidebar.slider(
                f.name, min_value=1, max_value=500, value=value
            )

        elif isinstance(value, float):
            kwargs[f.name] = st.sidebar.slider(
                f.name,
                min_value=0.0,
                max_value=5.0,
                value=float(value),
                step=0.05
            )

    return SimConfig(**kwargs)


cfg = build_config()

if st.session_state.sim is None or st.session_state.sim.cfg != cfg:
    st.session_state.sim = Simulation(cfg)
    st.session_state.cascade_id = 0
    st.session_state.verify_series = []
    st.session_state.true_series = []
    st.session_state.fake_series = []
    st.session_state.welfare_series = []
    st.session_state.true_total = 0.0
    st.session_state.fake_total = 0.0
    st.session_state.true_count = 0
    st.session_state.fake_count = 0
    st.session_state.welfare_total = 0.0

sim = st.session_state.sim

if "layout" not in st.session_state:
    st.session_state.layout = nx.spring_layout(sim.net.G, seed=42)


# ============================================================
# CONTROLS
# ============================================================

c1, c2 = st.columns(2)

if c1.button("⏭ Step"):
    st.session_state.running = False
    run_step = True
else:
    run_step = False

if c2.button("🔄 Reset"):
    st.session_state.sim = Simulation(cfg)
    st.session_state.cascade_id = 0
    st.session_state.verify_series = []
    st.session_state.true_series = []
    st.session_state.fake_series = []
    st.session_state.welfare_series = []
    st.session_state.true_total = 0.0
    st.session_state.fake_total = 0.0
    st.session_state.true_count = 0
    st.session_state.fake_count = 0
    st.session_state.welfare_total = 0.0
    st.session_state.running = False
    st.stop()


# ============================================================
# RUN STEP
# ============================================================

def run_single_step():
    stats = sim.run_single_cascade(st.session_state.cascade_id)
    st.session_state.cascade_id += 1

    # --- Verification Rate ---
    verify_rate = (
        stats["verifications"] / stats["receivers"]
        if stats["receivers"] > 0 else 0
    )

    # --- Running average ---
    if len(st.session_state.verify_series) == 0:
        running_avg = verify_rate
    else:
        prev_avg = st.session_state.verify_series[-1]
        t = st.session_state.cascade_id  # already incremented above
        running_avg = ((prev_avg * (t - 1)) + verify_rate) / t

    st.session_state.verify_series.append(running_avg)

    # --- Cascade Size ---
    cascade_size = stats["shares"] / cfg.num_agents

    if stats["is_fake"]:
        st.session_state.fake_total += cascade_size
        st.session_state.fake_count += 1
    else:
        st.session_state.true_total += cascade_size
        st.session_state.true_count += 1

    true_avg = (
        st.session_state.true_total / st.session_state.true_count
        if st.session_state.true_count > 0 else 0
    )
    fake_avg = (
        st.session_state.fake_total / st.session_state.fake_count
        if st.session_state.fake_count > 0 else 0
    )

    st.session_state.true_series.append(true_avg)
    st.session_state.fake_series.append(fake_avg)

    # --- Welfare running average ---
    st.session_state.welfare_total += stats["utility"]
    welfare_avg = (
        st.session_state.welfare_total / st.session_state.cascade_id
    )
    st.session_state.welfare_series.append(welfare_avg)


if st.session_state.running:
    run_single_step()
    time.sleep(0.05)
    st.rerun()

if run_step:
    run_single_step()


# ============================================================
# LAYOUT
# ============================================================

left, right = st.columns([3, 2])


# ============================================================
# LARGE METRIC PLOTS
# ============================================================

with left:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(st.session_state.verify_series, linewidth=2.5)
    axes[0, 0].set_title("Verification Rate", fontsize=16)
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].plot(st.session_state.true_series, linewidth=2.5)
    axes[0, 1].set_title("True Cascade Size (Running Avg)", fontsize=16)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(st.session_state.fake_series, linewidth=2.5)
    axes[1, 0].set_title("False Cascade Size (Running Avg)", fontsize=16)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(st.session_state.welfare_series, linewidth=2.5)
    axes[1, 1].set_title("Average Social Welfare", fontsize=16)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    st.pyplot(fig, width='stretch')


# ============================================================
# NETWORK WITH REPUTATION COLORING
# ============================================================

with right:
    fig_net, ax_net = plt.subplots(figsize=(10, 10))

    G = sim.net.G
    pos = st.session_state.layout

    reputations = np.array([
        G.nodes[n]["reputation"] for n in G.nodes()
    ])

    scale = np.std(reputations) + 1e-6
    compressed = np.tanh(reputations / (2 * scale))
    norm = plt.Normalize(vmin=-1, vmax=1)
    node_colors = plt.cm.RdBu(norm(compressed))

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5,
        node_size=60,
        ax=ax_net
    )

    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax_net)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_net, shrink=0.8)

    ax_net.set_axis_off()

    st.pyplot(fig_net, width="stretch")


# ============================================================
# PROGRESS
# ============================================================

max_cascades = 50
progress = st.session_state.cascade_id / max_cascades
st.progress(min(progress, 1.0))
