import networkx as nx
import numpy as np

from config import SimConfig

class SocialNetwork:
    def __init__(self, config: SimConfig, rng):
        self.cfg = config
        self.rng = rng

        self.G = nx.barabasi_albert_graph(
            n=config.num_agents,
            m=config.num_connections,
            seed=int(self.rng.integers(0, 10**9))
        )
        self.initialize_state()

    def initialize_state(self):        
        nx.set_node_attributes(self.G, 0.0, "reputation")
        base_global_belief = {
            'alpha': self.cfg.global_truth_prior_alpha,
            'beta': self.cfg.global_false_prior_beta
        }
        base_tech_tpr_belief = {
            "alpha": self.cfg.verification_detected_fakes_prior,
            "beta": self.cfg.verification_missed_fakes_prior,
        }
        base_tech_fpr_belief = {
            "alpha": self.cfg.verification_false_alarms_prior,
            "beta": self.cfg.verification_correct_rejections_prior
        }
        for node in self.G.nodes():
            self.G.nodes[node]['neighbor_trust'] = {}
            self.G.nodes[node]['true_belief'] = base_global_belief.copy()
            self.G.nodes[node]["tech_tpr_belief"] = base_tech_tpr_belief.copy()
            self.G.nodes[node]["tech_fpr_belief"] = base_tech_fpr_belief.copy()
            
        for u, v in self.G.edges():
            self._init_trust_one_way(u, v)
            self._init_trust_one_way(v, u)

    def _init_trust_one_way(self, observer, target):
        self.G.nodes[observer]['neighbor_trust'][target] = {
            'alpha': self.cfg.neighbor_truth_prior_alpha, 
            'beta': self.cfg.neighbor_false_prior_beta
        }

    def update_trust(self, observer, sender, sender_was_truthful: bool):
        trust_map = self.G.nodes[observer].get('neighbor_trust')
        if not trust_map or sender not in trust_map:
            return

        stats = trust_map[sender]
        if sender_was_truthful:
            stats['alpha'] += 1
        else:
            stats['beta'] += 1
            
    def update_global_belief(self, agent_id, item_was_fake: bool):
        stats = self.G.nodes[agent_id]['true_belief']
        if item_was_fake:
            stats['beta'] += 1
        else:
            stats['alpha'] += 1

    def rewire_network(self):
        edges_to_check = list(self.G.edges())
        num_edges = len(edges_to_check)
        if num_edges == 0:
            return

        all_nodes = list(self.G.nodes)
        reputations = np.array([self.G.nodes[n]['reputation'] for n in all_nodes], dtype=np.float32)
        min_rep = reputations.min()
        max_rep = reputations.max()
        weights = 1e-6 + (reputations - min_rep) / (max_rep - min_rep + 1e-6)
        total_weight = weights.sum()
        global_probs = weights / total_weight

        edge_rands = self.rng.random(num_edges, dtype=np.float32)

        for i, (u, v) in enumerate(edges_to_check):
            if not self.G.has_edge(u, v):
                continue

            trust_u_on_v = self.G.nodes[u]['neighbor_trust'].get(v)
            trust_v_on_u = self.G.nodes[v]['neighbor_trust'].get(u)
            if not trust_u_on_v or not trust_v_on_u:
                continue

            rel_u_v = trust_u_on_v['alpha'] / (trust_u_on_v['alpha'] + trust_u_on_v['beta'])
            rel_v_u = trust_v_on_u['alpha'] / (trust_v_on_u['alpha'] + trust_v_on_u['beta'])

            if rel_u_v < rel_v_u:
                weakest_trust_val = rel_u_v
                rewiring_node, passive_node = u, v
            else:
                weakest_trust_val = rel_v_u
                rewiring_node, passive_node = v, u

            if edge_rands[i] < (1.0 - weakest_trust_val):
                self.G.remove_edge(rewiring_node, passive_node)
                self.G.nodes[rewiring_node]['neighbor_trust'].pop(passive_node, None)
                self.G.nodes[passive_node]['neighbor_trust'].pop(rewiring_node, None)

                if self.G.degree(passive_node) == 0:
                    target = passive_node
                    avoid = rewiring_node
                else:
                    target = rewiring_node
                    avoid = passive_node

                self.rewire_node(target, all_nodes, global_probs, node_to_avoid=avoid)

    def rewire_node(self, u, all_nodes, global_probs, node_to_avoid=None):
        """
        Adds one new edge for node u.
        """
        invalid_set = set(self.G.neighbors(u))
        invalid_set.add(u)
        if node_to_avoid is not None:
            invalid_set.add(node_to_avoid)

        batch_size = 20
        candidate_indices = self.rng.choice(
            len(all_nodes), 
            size=batch_size, 
            p=global_probs, 
            replace=True
        )

        for idx in candidate_indices:
            candidate = all_nodes[idx]
            if candidate not in invalid_set:
                self._connect(u, candidate)
                return

        return

    def _connect(self, u, v):
        self.G.add_edge(u, v)
        self._init_trust_one_way(u, v)
        self._init_trust_one_way(v, u)