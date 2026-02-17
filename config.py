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
    
    # --- Verification Technology ---
    # Probability of correctly identifying a Fake as Fake.
    verification_accuracy: float = 0.95
    
    # --- Belief Priors ---
    neighbor_truth_prior_alpha: float = 1.0 
    neighbor_false_prior_beta: float = 0.2  # initial neighbor trust = 1 / (1 + 0.2) = 0.83
    global_truth_prior_alpha: float = 3.0
    global_false_prior_beta: float = 2.0 # initial global true prevalence = 3 / (3 + 2) = 0.6

    # --- Rewiring ---
    enable_rewiring: bool = True
    reputation_attachment_strength: float = 1.0 # Higher means more likely to connect to high-rep agents