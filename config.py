import dataclasses
from typing import Literal

@dataclasses.dataclass
class SimConfig:
    # --- Network Parameters ---
    num_agents: int = 1000
    network_type: Literal['watts_strogatz', 'barabasi_albert'] = 'watts_strogatz'
    # For Watts-Strogatz
    ws_k: int = 10           # Each node is connected to k nearest neighbors
    ws_p: float = 0.1        # Probability of rewiring
    # For Barabasi-Albert
    ba_m: int = 5            # Number of edges to attach from a new node
    
    # --- Trust & Bayesian Parameters ---
    initial_trust_prior: float = 2.0  # Weight of the prior (Stubbornness)
    baseline_trust: float = 0.9       # Initial probability that a neighbor is honest (0.0-1.0)
    
    # --- Agent Heterogeneity ---
    # Weight (w) of Global Prevalence vs Local History. 
    # 0.0 = Only trusts neighbors. 1.0 = Only trusts global stats.
    w_mean: float = 0.3   
    w_std: float = 0.02
    eta_w_update_rate: float = 0.05   # How fast an agent's w increases after being duped

    # CPT Parameters
    cpt_theta: float = 0.88
    cpt_lambda: float = 2.25
    cpt_gamma: float = 0.65

    # --- Economic/Utility Parameters ---
    R: float = 1                            # Reward for sharing TRUE news
    loss_aversion_ratio: float = 2.0        # How much more painful losing is compared to gaining
    debunking_premium_ratio: float = 1.5    # How much more rewarding debunking is compared to sharing truth
    C_mean: float = 2.0
    C_std_dev: float = 1.0

    # --- Deepfake & Environment Parameters ---
    global_prevalence: float = 0.3  # P(Global): Probability any random exogenous item is fake
    
    # --- Detection Technology (Noisy Oracle) ---
    # As Deepfakes improve, TPR should decrease (harder to catch fakes)
    # As Deepfakes improve, TNR may decrease (more false positives)
    true_positive_rate: float = 0.85 # Sensitivity: P(Detected Fake | Is Fake)
    true_negative_rate: float = 0.95 # Specificity: P(Detected True | Is True)