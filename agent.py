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
        # Avoid division by zero with small epsilon if needed, 
        # though priors usually prevent this.
        total_evidence = alpha + beta
        if total_evidence == 0:
            p_local = 0.5 # Maximum entropy if no data (shouldn't happen with priors)
        else:
            p_local = beta / total_evidence
            
        # 2. Weighted combination with Global Prevalence
        # p = (1-w)*Local + w*Global
        p_subjective = (1 - w) * p_local + w * config.global_prevalence
        
        return p_subjective

    @staticmethod
    def make_decision(p_fake: float, config: SimConfig) -> str:
        """
        Calculate Expected Utility for Share, Discard, Verify.
        Returns: 'SHARE', 'DISCARD', or 'VERIFY'
        """
        # Param shorthand
        R, L, D, C = config.R, config.L, config.D, config.C
        
        # 1. Utility of Sharing Immediately (Gambling)
        # If True (1-p): Gain R. If Fake (p): Lose L.
        u_share = (1 - p_fake) * R - (p_fake * L)
        
        # 2. Utility of Discarding (Safety)
        u_discard = 0.0
        
        # 3. Utility of Verifying (Investigation)
        # Agent assumes verification will reveal the truth (subjective view).
        # We subtract Cost C immediately.
        # Exp. Gain: If it turns out True (1-p)*R + If it turns out Fake (p)*D
        u_verify = -C + ((1 - p_fake) * R) + (p_fake * D)
        
        # Decision: Maximize Utility
        options = {'SHARE': u_share, 'DISCARD': u_discard, 'VERIFY': u_verify}
        best_choice = max(options, key=options.get)
        
        return best_choice

    @staticmethod
    def perform_verification(is_actually_fake: bool, config: SimConfig) -> bool:
        """
        Simulates the Noisy Oracle.
        Returns the DETECTED label (True = Detected as Fake, False = Detected as Real).
        """
        if is_actually_fake:
            # It is fake. Do we catch it?
            if np.random.random() < config.true_positive_rate:
                return True # Correctly identified as Fake
            else:
                return False # Missed it (False Negative)
        else:
            # It is real. Do we accidentally flag it?
            if np.random.random() < config.true_negative_rate:
                return False # Correctly identified as Real
            else:
                return True # False Positive (Flagged real news as fake)