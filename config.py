from dataclasses import dataclass

@dataclass
class SimConfig:

    random_seed: int = 42

    # --- Network ---
    num_agents: int = 50
    num_connections: int = 1
    
    # --- Simulation ---
    # Global probability a generated item is fake
    global_fake_prob: float = 0.5  
    
    # --- Payoffs & Costs ---
    verification_cost: float = 0.5
    rep_gain: float = 1.0       # Gain for sharing Truth
    rep_loss: float = 2.0       # Loss for sharing Fake
    rep_debunk: float = 0.5     # Gain for Debunking
    
    # --- Verification Technology ---
    # Probability of correctly identifying a Fake as Fake.
    verification_accuracy: float = 0.95
    
    # --- Belief Priors ---
    prior_alpha: float = 1.0 
    prior_beta: float = 0.2  # Starts slightly trusting
    initial_fake_belief: float = 0.3

    # --- Rewiring ---
    enable_rewiring: bool = False
    # If trust < threshold, break link
    rewire_trust_threshold: float = 0.3 
    # How often to check for rewiring (every N cascades)
    rewiring_interval: int = 1 