from dataclasses import dataclass

@dataclass
class SimConfig:

    random_seed: int = 42

    # Network 
    num_agents: int = 50
    num_connections: int = 1
    
    # Simulation 
    # Global probability a generated item is fake
    global_fake_prob: float = 0.5  
    
    # Payoffs & Costs 
    verification_cost: float = 0.5
    rep_gain: float = 1.0 # Gain for sharing Truth
    rep_loss: float = 2.0 # Loss for sharing Fake
    
    # Verification Technology Confusion Matrix
    verification_tpr: float = 0.90
    verification_fpr: float = 0.10
    
    # Belief Priors 
    neighbor_truth_prior_alpha: float = 1.0 
    neighbor_false_prior_beta: float = 0.2 # initial neighbor trust = 1 / (1 + 0.2) = 0.83
    global_truth_prior_alpha: float = 3.0
    global_false_prior_beta: float = 2.0 # initial global true prevalence = 3 / (3 + 2) = 0.6
    verification_detected_fakes_prior: float = 5.0
    verification_missed_fakes_prior: float = 1.0 # initial TPR = 5/6
    verification_false_alarms_prior: float = 1.0
    verification_correct_rejections_prior: float = 5.0 # initial FPR = 1/6

    # Rewiring 
    enable_rewiring: bool = True
    rewiring_interval: int = 10