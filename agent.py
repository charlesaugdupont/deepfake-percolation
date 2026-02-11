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
        """Bayesian update based on who sent the item."""
        p_true_global = 1.0 - global_fake_belief
        
        numerator = reliability * p_true_global
        denominator = (reliability * p_true_global) + ((1 - reliability) * global_fake_belief)
        
        if denominator == 0: return 0.5
        return numerator / denominator

    @staticmethod
    def decide(p_true: float, reach: int, cfg: SimConfig):
        """
        Calculates EU for Share, Discard, Verify.
        Includes risk assessment for Noisy Verification.
        """
        k = reach
        ver_acc = cfg.verification_accuracy
        
        # --- Pre-calculate Utility of Outcomes ---
        
        # 1. We Share, and it is TRUE
        u_share_if_true = cfg.rep_gain * k
        
        # 2. We Share, and it is FAKE
        u_share_if_fake = -cfg.rep_loss * k
        
        # 3. We Debunk (Only happens if we find a Fake)
        u_debunk = cfg.rep_debunk * k

        # --- Evaluate Actions ---
        
        # Action: DISCARD
        u_discard = 0.0
        
        # Action: SHARE (Blindly, no verification)
        u_share = (p_true * u_share_if_true) + ((1 - p_true) * u_share_if_fake)
        
        # Action: VERIFY
        u_verify_cost = -cfg.verification_cost
        
        # Branch A: Item is actually TRUE (prob p_true)
        exp_u_given_true = u_share_if_true
        
        # Branch B: Item is actually FAKE (prob 1-p_true)
        # We verify. Two sub-outcomes:
        #   1. Success: We detect Fake -> Debunk.
        #   2. Failure: We think it's True -> Share -> GET PUNISHED.
        exp_u_given_fake = (ver_acc * u_debunk) + ((1 - ver_acc) * u_share_if_fake)
        
        u_verify = u_verify_cost + (p_true * exp_u_given_true) + ((1 - p_true) * exp_u_given_fake)
        
        # --- Maximize ---
        options = {'DISCARD': u_discard, 'SHARE': u_share, 'VERIFY': u_verify}
        best_action = max(options, key=options.get)
        
        return best_action