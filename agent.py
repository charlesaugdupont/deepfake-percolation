from config import SimConfig

class AgentLogic:
    @staticmethod
    def compute_beta_mean(alpha: float, beta: float) -> float:
        """Returns mean of Beta distribution"""
        return alpha / (alpha + beta)

    @staticmethod
    def calculate_posterior_truth_prob(
        reliability: float, 
        global_true_belief: float
    ) -> float:
        """Bayesian update based on who sent the item."""
        p_false_global = 1.0 - global_true_belief
        numerator = reliability * global_true_belief
        denominator = (reliability * global_true_belief) + ((1 - reliability) * p_false_global)
        return numerator / denominator

    @staticmethod
    def decide(p_true: float, reach: int, est_tpr:float, est_fpr:float, cfg: SimConfig):
        """
        Calculates EU for Share, Discard, Verify.
        Includes risk assessment for Noisy Verification.
        """
        k = reach
        
        # Pre-calculate Utility of Outcomes
        
        # 1. We Share, and it is TRUE
        u_share_if_true = cfg.rep_gain * k
        
        # 2. We Share, and it is FAKE
        u_share_if_fake = -cfg.rep_loss * k
        
        # Action: DISCARD
        u_discard = 0.0
        
        # Action: SHARE (Blindly, no verification)
        u_share = (p_true * u_share_if_true) + ((1 - p_true) * u_share_if_fake)
        
        # Action: VERIFY

        # item is actually real
        eu_given_real = (1 - est_fpr) * (cfg.rep_gain * reach)

        # item is actually fake
        eu_given_fake = (1 - est_tpr) * (-cfg.rep_loss * reach)

        u_verify = -cfg.verification_cost + (p_true * eu_given_real) + ((1 - p_true) * eu_given_fake)

        # Find the action with the highest expected utility
        if u_share > u_discard:
            if u_share > u_verify:
                return 'SHARE'
            else:
                return 'VERIFY'
        else:
            if u_discard >= u_verify:
                return 'DISCARD'
            else:
                return 'VERIFY'