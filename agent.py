import numpy as np
from config import SimConfig

class AgentLogic:
    """
    Stateless logic for agent decision making. 
    State is stored in the NetworkX graph, passed here for calculation.
    """

    @staticmethod
    def estimate_p_fake(alpha: float, beta: float, w: float, config: SimConfig) -> float:
        """
        Calculate subjective probability that an item is Fake.
        Combines Local Bayesian estimate with Global prior.
        """
        # 1. Local Estimate (Mean of Beta Distribution)
        total_evidence = alpha + beta
        if total_evidence == 0:
            p_local = 0.5
        else:
            p_local = beta / total_evidence
            
        # 2. Weighted combination with Global Prevalence
        p_subjective = (1 - w) * p_local + w * config.global_prevalence
        
        return p_subjective
    
    @staticmethod
    def _cpt_value_function(outcome, config: SimConfig):
        """Applies the CPT value function to a utility outcome."""
        if outcome >= 0:
            return outcome ** config.cpt_theta
        else:
            return -config.cpt_lambda * ((-outcome) ** config.cpt_theta)

    @staticmethod
    def _cpt_probability_weighting(p, config: SimConfig):
        """Applies the Tversky-Kahneman probability weighting function."""
        g = config.cpt_gamma
        return (p**g) / ((p**g + (1-p)**g)**(1/g))

    @staticmethod
    def make_cpt_decision(p_fake: float, item_cost: float, config: SimConfig) -> str:
        """Calculates CPT value for each choice."""
        # Derived L and D (actual payoff values, not just ratios)
        L = config.R * config.loss_aversion_ratio
        D = config.R * config.debunking_premium_ratio
        
        # Weight the probabilities
        w_p_fake = AgentLogic._cpt_probability_weighting(p_fake, config)
        w_p_true = AgentLogic._cpt_probability_weighting(1 - p_fake, config)

        # 1. CPT Value of Sharing
        val_share = (w_p_true * AgentLogic._cpt_value_function(config.R, config)) + \
                    (w_p_fake * AgentLogic._cpt_value_function(-L, config))

        # 2. CPT Value of Discarding
        val_discard = 0.0

        # 3. CPT Value of Verifying
        val_verify = AgentLogic._cpt_value_function(-item_cost, config) + \
                     (w_p_true * AgentLogic._cpt_value_function(config.R, config)) + \
                     (w_p_fake * AgentLogic._cpt_value_function(D, config))

        options = {'SHARE': val_share, 'DISCARD': val_discard, 'VERIFY': val_verify}
        return max(options, key=options.get)

    @staticmethod
    def perform_verification(is_actually_fake: bool, config: SimConfig) -> bool:
        """
        Simulates the Noisy Oracle.
        Returns the DETECTED label (True = Detected as Fake, False = Detected as Real).
        """
        if is_actually_fake:
            if np.random.random() < config.true_positive_rate:
                return True  # Correctly identified as Fake
            else:
                return False  # Missed it (False Negative)
        else:
            if np.random.random() < config.true_negative_rate:
                return False  # Correctly identified as Real
            else:
                return True  # False Positive