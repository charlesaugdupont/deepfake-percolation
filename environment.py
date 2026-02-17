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
        base_belief = {
            'alpha': self.cfg.global_truth_prior_alpha,
            'beta': self.cfg.global_false_prior_beta
        }
        
        for node in self.G.nodes():
            self.G.nodes[node]['true_belief'] = base_belief.copy()
            self.G.nodes[node]['neighbor_trust'] = {}
            
        # initialize trust
        for u, v in self.G.edges():
            self._init_trust_one_way(u, v)
            self._init_trust_one_way(v, u)

    def _init_trust_one_way(self, observer, target):
        self.G.nodes[observer]['neighbor_trust'][target] = {
            'alpha': self.cfg.neighbor_truth_prior_alpha, 
            'beta': self.cfg.neighbor_false_prior_beta
        }

    def get_reach(self, node_id, visited_set):
        """
        Returns marginal reach (count of neighbors who are NOT in visited_set)
        """
        return sum(1 for n in self.G.neighbors(node_id) if n not in visited_set)

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

        edge_rands = self.rng.random(num_edges, dtype=np.float32)
        nodes = self.G.nodes

        for i, (u, v) in enumerate(edges_to_check):
            if not self.G.has_edge(u, v):
                continue

            trust_u_on_v = nodes[u]['neighbor_trust'].get(v)
            trust_v_on_u = nodes[v]['neighbor_trust'].get(u)
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
                nodes[rewiring_node]['neighbor_trust'].pop(passive_node, None)
                nodes[passive_node]['neighbor_trust'].pop(rewiring_node, None)

                if self.G.degree(passive_node) == 0:
                    self.rewire_node(passive_node, nodes, node_to_avoid=rewiring_node)
                else:
                    self.rewire_node(rewiring_node, nodes, node_to_avoid=passive_node)

    def rewire_node(self, u, nodes, node_to_avoid=None):
        """
        Adds one new edge for node u.
        """
        candidates = list(set(self.G.nodes) - set(self.G.neighbors(u)) - {u} - {node_to_avoid})
        if len(candidates) == 0:
            return
        reputation = np.array([nodes[n]['reputation'] for n in candidates], dtype=np.float32)

        # because reputation can be negative, we do min-max normalization to get non-negative weights
        min_rep = reputation.min()
        max_rep = reputation.max()
        weights = 1e-6 + (reputation - min_rep) / (max_rep - min_rep + 1e-6)  # add small epsilon to avoid division by zero

        total_weight = weights.sum()
        probs = weights / total_weight

        choice = self.rng.choice(len(candidates), p=probs)        
        v = candidates[choice]
        self.G.add_edge(u, v)
        self._init_trust_one_way(u, v)
        self._init_trust_one_way(v, u)