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
        # get all edges as a list
        edges = list(self.G.edges())
        num_edges = len(edges)
        if num_edges == 0:
            return

        edge_rands = self.rng.random(num_edges, dtype=np.float32)
        edges_to_remove = []
        nodes_seeking_new_connection = []
        nodes_data = self.G.nodes
        
        # iterate over existing edges
        for i, (u, v) in enumerate(edges):
            trust_u = nodes_data[u]['neighbor_trust'].get(v)
            trust_v = nodes_data[v]['neighbor_trust'].get(u)
            if not trust_u or not trust_v:
                continue

            # calculate trust
            rel_u_v = trust_u['alpha'] / (trust_u['alpha'] + trust_u['beta'])
            rel_v_u = trust_v['alpha'] / (trust_v['alpha'] + trust_v['beta'])

            if rel_u_v < rel_v_u:
                weakest_val = rel_u_v
                node_to_rewire = u
            else:
                weakest_val = rel_v_u
                node_to_rewire = v
            
            # probabilistic cutting
            if edge_rands[i] < (1.0 - weakest_val):
                edges_to_remove.append((u, v))
                nodes_seeking_new_connection.append(node_to_rewire)

        # apply cuts
        for u, v in edges_to_remove:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
                # clean up trust dicts to save memory
                nodes_data[u]['neighbor_trust'].pop(v, None)
                nodes_data[v]['neighbor_trust'].pop(u, None)

        # connect isolated nodes
        deg = dict(self.G.degree())
        for n, d in deg.items():
            if d == 0:
                nodes_seeking_new_connection.append(n)

        if not nodes_seeking_new_connection:
            return
        
        all_nodes = list(self.G.nodes())
        n_total = len(all_nodes)
        
        # extract reputations
        reps = np.array([nodes_data[n]['reputation'] for n in all_nodes])
        
        # normalize
        if reps.std() > 1e-9:
            reps_norm = (reps - reps.mean()) / reps.std()
        else:
            reps_norm = np.zeros_like(reps)
            
        weights = np.exp(self.cfg.reputation_attachment_strength * reps_norm)
        probs = weights / weights.sum()

        # rejection sampling
        num_needed = len(nodes_seeking_new_connection)
        
        # draw indices instead of node objects for speed
        buffer_size = int(5 * num_needed)
        target_indices = self.rng.choice(
            n_total, 
            size=buffer_size,
            p=probs, 
            replace=True
        )
        
        # rewire
        idx = 0
        for u in nodes_seeking_new_connection:
            while True:
                if idx >= len(target_indices):
                    extra_indices = self.rng.choice(
                        n_total, 
                        size=buffer_size, 
                        p=probs, 
                        replace=True
                    )
                    target_indices = np.concatenate([target_indices, extra_indices])
                    idx = 0

                target_idx = target_indices[idx]
                idx += 1
                v = all_nodes[target_idx]
                if u == v: 
                    continue # skip self-loop
                elif self.G.has_edge(u, v):
                    continue # skip existing edges
                else:
                    break
            # connect and initialize trust
            self.G.add_edge(u, v)
            self._init_trust_one_way(u, v)
            self._init_trust_one_way(v, u)