import numpy as np
from config import SimConfig

class AgentLogic:
    @staticmethod
    def get_neighbor_reliability(alpha: float, beta: float) -> float:
        """Returns mean of Beta distribution: P(Neighbor is Truthful)."""
        return alpha / (alpha + beta)

    @staticmethod
    def calculate_posterior_truth_prob(
        reliability: float, 
        global_fake_belief: float
    ) -> float:
        """
        Bayesian update:
        P(True | Received) = (P(Received | True) * P(True)) / P(Received)
        Assuming reliable neighbor sends True, unreliable sends random/noise.
        Simplified strictly for this model:
        """
        p_true_global = 1.0 - global_fake_belief
        
        numerator = reliability * p_true_global
        denominator = (reliability * p_true_global) + ((1 - reliability) * global_fake_belief)
        
        # Avoid division by zero
        if denominator == 0: return 0.5
        return numerator / denominator

    @staticmethod
    def decide(p_true: float, reach: int, cfg: SimConfig):
        """
        Calculates U(Share), U(Discard), U(Verify) based on Weighted Additive Utility.
        Returns: Action string and Estimated Utility
        """
        w = cfg.self_interest_weight
        k = reach
        
        # 1. Utility of DISCARD
        u_discard = 0.0
        
        # 2. Utility of SHARE (Immediate)
        # Rep component: p * Gain - (1-p) * Loss
        u_share_rep = (p_true * cfg.rep_gain * k) - ((1 - p_true) * cfg.rep_loss * k)
        # Global component: p * Reach - (1-p) * Reach (Simple: Truth helps, Fake hurts)
        u_share_global = (p_true * k) - ((1 - p_true) * k)
        
        u_share = w * u_share_rep + (1 - w) * u_share_global
        
        # 3. Utility of VERIFY
        # Cost is paid upfront
        # If True (prob p_true): We will Share.
        # If Fake (prob 1-p_true): We will Debunk.
        
        # Value given it turns out TRUE (We share)
        val_if_true_rep = cfg.rep_gain * k
        val_if_true_glob = k
        u_post_verify_true = w * val_if_true_rep + (1 - w) * val_if_true_glob
        
        # Value given it turns out FAKE (We debunk)
        val_if_fake_rep = cfg.rep_debunk * k
        val_if_fake_glob = k # We saved them!
        u_post_verify_fake = w * val_if_fake_rep + (1 - w) * val_if_fake_glob
        
        u_verify = -cfg.verification_cost + \
                   (p_true * u_post_verify_true) + \
                   ((1 - p_true) * u_post_verify_fake)
        
        # Maximize
        options = {'DISCARD': u_discard, 'SHARE': u_share, 'VERIFY': u_verify}
        best_action = max(options, key=options.get)
        
        return best_action