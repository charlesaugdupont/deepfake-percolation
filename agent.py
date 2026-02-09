import numpy as np
from config import SimConfig

class AgentLogic:
    @staticmethod
    def estimate_p_fake(alpha: float, beta: float, w: float, p_global: float) -> float:
        total = alpha + beta
        p_local = 0.5 if total == 0 else beta / total
        return (1 - w) * p_local + w * p_global

    @staticmethod
    def make_decision(p_fake: float, config: SimConfig) -> str:
        R, L, D, C = config.R, config.L, config.D, config.C
        tpr, tnr = config.true_positive_rate, config.true_negative_rate

        u_share = (1 - p_fake) * R - p_fake * L
        u_discard = 0.0

        u_verify = -C
        u_verify += p_fake * (tpr * D + (1 - tpr) * (-L))
        u_verify += (1 - p_fake) * (tnr * R)

        return max(
            {'SHARE': u_share, 'DISCARD': u_discard, 'VERIFY': u_verify},
            key=lambda k: {'SHARE': u_share, 'DISCARD': u_discard, 'VERIFY': u_verify}[k]
        )

    @staticmethod
    def perform_verification(is_actually_fake: bool, config: SimConfig) -> bool:
        if is_actually_fake:
            return np.random.random() < config.true_positive_rate
        return not (np.random.random() < config.true_negative_rate)
