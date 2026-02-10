from dataclasses import dataclass

@dataclass
class SimConfig:
    # --- Network ---
    num_agents: int = 500
    num_connections: int = 2
    
    # --- Simulation ---
    num_cascades: int = 100
    
    # --- Information ---
    global_fake_prob: float = 0.5  # Actual probability a new item is fake
    
    # --- Payoffs & Costs ---
    verification_cost: float = 0.5  # c_v
    
    rep_gain: float = 1.0       # g: Gain for sharing Truth
    rep_loss: float = 2.0       # l: Loss for sharing Fake
    rep_debunk: float = 1.0     # d: Gain for Debunking Fake (New!)
    
    # --- Agent Psychology ---
    # w: 1.0 = Pure Reputation (Selfish), 0.0 = Pure Truth (Altruist)
    self_interest_weight: float = 0.7
    
    # --- Belief Priors ---
    # Initial belief about neighbor reliability (Beta distribution)
    prior_alpha: float = 1.0 
    prior_beta: float = 0.2  # Starts slightly trusting
    
    # Initial belief about global fake rate
    initial_fake_belief: float = 0.3