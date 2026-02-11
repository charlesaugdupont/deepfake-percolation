import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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
# SIDEBAR — CLEAN PARAM CONTROLS
# ============================================================

st.sidebar.title("Parameters")
random_seed = st.sidebar.number_input("Random Seed", 0, 1_000_000, 42, 1)

# --- Network structure ---
st.sidebar.divider()
st.sidebar.markdown("### Network / Global")
N = st.sidebar.slider("N", 10, 500, 50, step=5)
m = st.sidebar.slider("m", 1, 20, 1)
global_fake_prob = st.sidebar.slider("Fake Probability", 0.0, 1.0, 0.5, 0.02)

# --- Payoffs ---
st.sidebar.divider()
st.sidebar.markdown("### Payoffs")
C = st.sidebar.slider("Verification Cost", 0.0, 5.0, 1.0, 0.05)
G = st.sidebar.slider("Reputation Gain", 0.0, 5.0, 1.0, 0.05)
L = st.sidebar.slider("Reputation Loss", 0.0, 5.0, 2.0, 0.05)
D = st.sidebar.slider("Debunk Reward", 0.0, 5.0, 1.5, 0.05)

# --- Verification Technology ---
st.sidebar.divider()
st.sidebar.markdown("### Verification Technology")
verification_accuracy = st.sidebar.slider("Verification Accuracy", 0.0, 1.0, 0.95, 0.01)

# --- Agent Psychology ---
st.sidebar.divider()
st.sidebar.markdown("### Agent Psychology")
self_interest_weight = st.sidebar.slider("Self-Interest Weight", 0.0, 1.0, 0.5, 0.02)
prior_alpha = st.sidebar.slider("Prior Neighbor Trust (True Weight)", 0.1, 1.0, 1.0, 0.1)
prior_beta = st.sidebar.slider("Prior Neighbor Trust (False Weight)", 0.1, 1.0, 0.2, 0.1)
initial_fake_belief = st.sidebar.slider("Initial Fake Prevalence Belief", 0.0, 1.0, 0.3, 0.02)

# --- Rewiring ---
st.sidebar.divider()
st.sidebar.markdown("### Rewiring")
rewiring = st.sidebar.checkbox("Enable Rewiring", value=False)
rewiring_interval = st.sidebar.slider("Rewiring Interval", 1, 10, 1)
rewire_trust_threshold = st.sidebar.slider("Rewiring Trust Threshold", 0.0, 1.0, 0.3, 0.02)

# ============================================================
# BUILD CONFIG
# ============================================================

cfg = SimConfig(
    num_agents=N,
    num_connections=m,
    global_fake_prob=global_fake_prob,
    verification_cost=C,
    rep_gain=G,
    rep_loss=L,
    rep_debunk=D,
    enable_rewiring=rewiring,
    rewiring_interval=rewiring_interval,
    rewire_trust_threshold=rewire_trust_threshold,
    verification_accuracy=verification_accuracy,
    self_interest_weight=self_interest_weight,
    prior_alpha=prior_alpha,
    prior_beta=prior_beta,
    initial_fake_belief=initial_fake_belief,
    random_seed=random_seed
)

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
    st.session_state.layout = nx.spring_layout(
        st.session_state.sim.net.G,
        seed=cfg.random_seed
    )

sim = st.session_state.sim


# ============================================================
# CONTROLS
# ============================================================

c1, c2, c3 = st.columns(3)

# --- Step ---
if c1.button("⏭ Step"):
    run_step = True
else:
    run_step = False

# --- Run Full Simulation ---
if c2.button("▶ Run Full Simulation"):
    run_full = True
else:
    run_full = False

# --- Reset ---
if c3.button("🔄 Reset"):
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
    st.stop()


# ============================================================
# RUN LOGIC
# ============================================================

MAX_CASCADES = 100

def run_single_step():
    stats = sim.run_single_cascade()
    st.session_state.cascade_id += 1

    # --- Verification running average ---
    verify_rate = (
        stats["verifications"] / stats["receivers"]
        if stats["receivers"] > 0 else 0
    )

    t = st.session_state.cascade_id

    if t == 1:
        running_avg = verify_rate
    else:
        prev_avg = st.session_state.verify_series[-1]
        running_avg = ((prev_avg * (t - 1)) + verify_rate) / t

    st.session_state.verify_series.append(running_avg)

    # --- Cascade Size ---
    cascade_size = stats["shares"] / N

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
    welfare_avg = st.session_state.welfare_total / t
    st.session_state.welfare_series.append(welfare_avg)

def run_full_simulation(max_cascades):
    remaining = max_cascades - st.session_state.cascade_id
    for _ in range(remaining):
        run_single_step()

if run_step:
    run_single_step()

if run_full:
    run_full_simulation(MAX_CASCADES)


# ============================================================
# LAYOUT
# ============================================================

left, right = st.columns([3, 2])


# ============================================================
# LARGE METRIC PLOTS
# ============================================================

with left:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(st.session_state.true_series, linewidth=2.5, color="seagreen")
    axes[0, 0].set_title("True Cascade Size (Running Avg)", fontsize=16)
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].plot(st.session_state.fake_series, linewidth=2.5, color="crimson")
    axes[0, 1].set_title("False Cascade Size (Running Avg)", fontsize=16)
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(st.session_state.verify_series, linewidth=2.5)
    axes[1, 0].set_title("Verification Rate (Running Avg)", fontsize=16)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(st.session_state.welfare_series, linewidth=2.5)
    axes[1, 1].set_title("Average Social Welfare", fontsize=16)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=12)

    plt.tight_layout()
    st.pyplot(fig, width="stretch")


# ============================================================
# NETWORK WITH REPUTATION COLORING
# ============================================================

with right:
    fig_net, ax_net = plt.subplots(figsize=(10, 10))

    G_net = sim.net.G
    pos = st.session_state.layout

    reputations = np.array([
        G_net.nodes[n]["reputation"] for n in G_net.nodes()
    ])

    scale = np.std(reputations) + 1e-6
    compressed = np.tanh(reputations / (2 * scale))

    norm = plt.Normalize(vmin=-1, vmax=1)
    node_colors = plt.cm.RdBu(norm(compressed))

    nx.draw_networkx_nodes(
        G_net,
        pos,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.5,
        node_size=60,
        ax=ax_net
    )

    nx.draw_networkx_edges(G_net, pos, alpha=0.2, ax=ax_net)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.RdBu)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_net, shrink=0.8)

    ax_net.set_axis_off()

    st.pyplot(fig_net, width="stretch")


# ============================================================
# PROGRESS
# ============================================================

progress = st.session_state.cascade_id / MAX_CASCADES
st.progress(min(progress, 1.0))
