import numpy as np
from config import SimConfig

class AgentLogic:
    """
    Stateless logic for agent decision making.
    """

    @staticmethod
    def estimate_p_fake(alpha: float, beta: float, w: float, config: SimConfig) -> float:
        total = alpha + beta
        p_local = 0.5 if total == 0 else beta / total
        return (1 - w) * p_local + w * config.global_prevalence

    @staticmethod
    def make_decision(p_fake: float, config: SimConfig) -> str:
        R, L, D, C = config.R, config.L, config.D, config.C
        tpr, tnr = config.true_positive_rate, config.true_negative_rate

        # SHARE (blind)
        u_share = (1 - p_fake) * R - p_fake * L

        # DISCARD
        u_discard = 0.0

        # VERIFY (correct expected value)
        u_verify = -C

        # If item is fake
        u_verify += p_fake * (
            tpr * D              # correctly debunk
            + (1 - tpr) * (-L)   # miss fake and share it
        )

        # If item is real
        u_verify += (1 - p_fake) * (
            tnr * R              # correctly share truth
            + (1 - tnr) * 0.0    # false positive: discard truth
        )

        return max(
            {'SHARE': u_share, 'DISCARD': u_discard, 'VERIFY': u_verify},
            key=lambda k: {'SHARE': u_share, 'DISCARD': u_discard, 'VERIFY': u_verify}[k]
        )

    @staticmethod
    def perform_verification(is_actually_fake: bool, config: SimConfig) -> bool:
        if is_actually_fake:
            return np.random.random() < config.true_positive_rate
        else:
            return not (np.random.random() < config.true_negative_rate)
