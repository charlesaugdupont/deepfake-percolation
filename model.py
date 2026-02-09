import networkx as nx
import numpy as np
import collections
from config import SimConfig
from agent import AgentLogic

class DeepfakePercolationModel:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.G = self._build_network()
        self._initialize_agent_states()

    def _build_network(self):
        if self.cfg.network_type == 'watts_strogatz':
            return nx.watts_strogatz_graph(
                self.cfg.num_agents, self.cfg.ws_k, self.cfg.ws_p
            )
        return nx.barabasi_albert_graph(self.cfg.num_agents, self.cfg.ba_m)

    def _initialize_agent_states(self):
        # Node-level beliefs
        for n in self.G.nodes():
            w = np.clip(np.random.normal(self.cfg.w_mean, self.cfg.w_std), 0, 1)
            self.G.nodes[n]['w'] = w

            # Global belief priors (same for all, but could be heterogeneous)
            prior = self.cfg.initial_trust_prior
            self.G.nodes[n]['g_alpha'] = prior * (1 - self.cfg.global_prevalence)
            self.G.nodes[n]['g_beta'] = prior * self.cfg.global_prevalence

        # Edge-level trust
        alpha0 = self.cfg.initial_trust_prior * self.cfg.baseline_trust
        beta0 = self.cfg.initial_trust_prior * (1 - self.cfg.baseline_trust)

        for u, v in self.G.edges():
            self.G[u][v]['trust'] = {
                u: {'alpha': alpha0, 'beta': beta0},
                v: {'alpha': alpha0, 'beta': beta0}
            }

    # ---- Trust helpers ----

    def get_trust(self, observer, sender):
        return self.G[observer][sender]['trust'][observer]

    def update_local_trust(self, observer, sender, sender_correct: bool):
        trust = self.get_trust(observer, sender)
        if sender_correct:
            trust['alpha'] += 1
        else:
            trust['beta'] += 1

    # ---- Global belief helpers ----

    def update_global_belief(self, agent, is_fake: bool):
        if is_fake:
            self.G.nodes[agent]['g_beta'] += 1
        else:
            self.G.nodes[agent]['g_alpha'] += 1

    def get_p_global(self, agent):
        a = self.G.nodes[agent]['g_alpha']
        b = self.G.nodes[agent]['g_beta']
        return b / (a + b)

    # ---- Cascade ----

    def run_cascade(self, source_node: int, is_ground_truth_fake: bool):
        queue = collections.deque()
        received = {source_node}

        for n in self.G.neighbors(source_node):
            received.add(n)
            queue.append((n, source_node))

        stats = {
            'total_shared': 0,
            'utility': 0.0,
            'num_received': 0,
            'num_verified': 0
        }

        while queue:
            current, sender = queue.popleft()
            stats['num_received'] += 1

            trust = self.get_trust(current, sender)
            w = self.G.nodes[current]['w']
            p_global = self.get_p_global(current)

            p_fake = AgentLogic.estimate_p_fake(
                trust['alpha'], trust['beta'], w, p_global
            )

            action = AgentLogic.make_decision(p_fake, self.cfg)
            detected_as_fake = False
            shared = False

            if action == 'VERIFY':
                stats['num_verified'] += 1
                detected_as_fake = AgentLogic.perform_verification(
                    is_ground_truth_fake, self.cfg
                )

                # Update own beliefs
                self.update_global_belief(current, is_ground_truth_fake)

                sender_correct = (detected_as_fake == is_ground_truth_fake)
                self.update_local_trust(current, sender, sender_correct)

                if detected_as_fake:
                    # ---- Debunk spreads ----
                    for nbr in self.G.neighbors(current):
                        self.update_global_belief(nbr, True)
                else:
                    shared = True

            elif action == 'SHARE':
                shared = True

            payoff = self.apply_payoff(action, is_ground_truth_fake, detected_as_fake)
            stats['utility'] += payoff

            if shared:
                stats['total_shared'] += 1
                for n in self.G.neighbors(current):
                    if n not in received:
                        received.add(n)
                        queue.append((n, current))

        return stats

    def apply_payoff(self, action, is_fake, detected):
        cfg = self.cfg

        if action == 'DISCARD':
            return 0.0

        if action == 'SHARE':
            return -cfg.L if is_fake else cfg.R

        if action == 'VERIFY':
            payoff = -cfg.C
            if detected:
                payoff += cfg.D if is_fake else 0.0
            else:
                payoff += -cfg.L if is_fake else cfg.R
            return payoff
