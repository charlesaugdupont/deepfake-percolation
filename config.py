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
    w_mean: float = 0.2     
    w_std: float = 0.05
    
    # --- Economic/Utility Parameters ---
    R: float = 10.0      # Reward for sharing TRUE news
    L: float = 20.0      # Penalty for sharing FAKE news (Reputation loss)
    D: float = 15.0      # Reward for detecting/debunking FAKE news
    C: float = 2.0       # Cost of Verification (The main control parameter)
    
    # --- Deepfake & Environment Parameters ---
    global_prevalence: float = 0.3  # P(Global): Probability any random exogenous item is fake
    
    # --- Detection Technology (Noisy Oracle) ---
    # As Deepfakes improve, TPR decreases (harder to catch fakes)
    # As Deepfakes improve, TNR might decrease (more false positives)
    true_positive_rate: float = 0.85 # Sensitivity: P(Detected Fake | Is Fake)
    true_negative_rate: float = 0.95 # Specificity: P(Detected True | Is True)