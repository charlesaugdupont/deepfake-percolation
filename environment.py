import networkx as nx
import numpy as np

from config import SimConfig
from agent import AgentLogic

class SocialNetwork:
    def __init__(self, config: SimConfig, rng):
        self.cfg = config
        self.rng = rng

        self.G = nx.barabasi_albert_graph(
            n=config.num_agents,
            m=config.num_connections,
            seed=int(self.rng.integers(0, 10**9)) # generic rng to int seed
        )
        self.initialize_state()

    def initialize_state(self):
        # OPTIMIZATION: Use bulk attribute setting instead of iterating
        # This is much faster in NetworkX internal C loops
        
        # 1. Base attributes
        nx.set_node_attributes(self.G, 0.0, "reputation")
        
        # 2. Complex objects (dicts) still need generation, but we can do it cleanly
        # Shared belief object structure
        base_belief = {
            'alpha': self.cfg.global_truth_prior_alpha,
            'beta': self.cfg.global_false_prior_beta
        }
        
        for node in self.G.nodes():
            # Copy is essential so nodes don't share the same reference
            self.G.nodes[node]['true_belief'] = base_belief.copy()
            self.G.nodes[node]['neighbor_trust'] = {}
            
        # 3. Initialize Trust (Iterate edges, not nodes -> neighbors)
        for u, v in self.G.edges():
            self._init_trust_one_way(u, v)
            self._init_trust_one_way(v, u)

    def _init_trust_one_way(self, observer, target):
        """Optimized internal helper"""
        self.G.nodes[observer]['neighbor_trust'][target] = {
            'alpha': self.cfg.neighbor_truth_prior_alpha, 
            'beta': self.cfg.neighbor_false_prior_beta
        }

    def get_reach(self, node_id, visited_set):
        """
        Marginal Reach: Count of neighbors who are NOT in visited_set.
        OPTIMIZATION: Generator expression avoids creating a list in memory.
        """
        # sum(1 for ...) is faster and memory efficient for counting
        return sum(1 for n in self.G.neighbors(node_id) if n not in visited_set)

    def update_trust(self, observer, sender, sender_was_truthful: bool):
        # Direct dict lookup is fast enough
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
        """
        Fully vectorized rewiring.
        1. Identify weak edges to cut.
        2. Calculate global reputation probabilities ONCE.
        3. Batch sample new connections.
        """

        # 1. Get all edges as a list
        edges = list(self.G.edges())
        num_edges = len(edges)
        if num_edges == 0:
            return

        edge_rands = self.rng.random(num_edges, dtype=np.float32)

        edges_to_remove = []
        nodes_seeking_new_connection = []
        nodes_data = self.G.nodes
        
        # iterate existing edges
        for i, (u, v) in enumerate(edges):
            trust_u = nodes_data[u]['neighbor_trust'].get(v)
            trust_v = nodes_data[v]['neighbor_trust'].get(u)
            if not trust_u or not trust_v:
                continue

            # Calculate Reliability
            # Inline the math if AgentLogic overhead is high, but method call usually fine
            rel_u_v = trust_u['alpha'] / (trust_u['alpha'] + trust_u['beta'])
            rel_v_u = trust_v['alpha'] / (trust_v['alpha'] + trust_v['beta'])

            # Logic: If the *weaker* link is below random threshold, cut.
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

        # Apply Cuts
        for u, v in edges_to_remove:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
                # Cleanup trust dicts to save memory
                nodes_data[u]['neighbor_trust'].pop(v, None)
                nodes_data[v]['neighbor_trust'].pop(u, None)

        # Add Isolated Nodes (Degree 0)
        deg = dict(self.G.degree())
        for n, d in deg.items():
            if d == 0:
                nodes_seeking_new_connection.append(n)

        # Return early if nothing to do
        if not nodes_seeking_new_connection:
            return

        # --- Phase 2: Vectorized Selection ---
        
        # 1. Prepare Global Arrays ONCE (not per node)
        all_nodes = list(self.G.nodes())
        n_total = len(all_nodes)
        
        # Extract reputations efficiently
        reps = np.array([nodes_data[n]['reputation'] for n in all_nodes])
        
        # Normalize / Softmax
        if reps.std() > 1e-9:
            reps_norm = (reps - reps.mean()) / reps.std()
        else:
            reps_norm = np.zeros_like(reps)
            
        weights = np.exp(self.cfg.reputation_attachment_strength * reps_norm)
        probs = weights / weights.sum()

        # 2. Batch Sample Targets
        # We need k targets. We sample k * 1.5 to account for self-loops/collisions
        # which we will discard. This is "Rejection Sampling".
        num_needed = len(nodes_seeking_new_connection)
        
        # Draw indices instead of node objects for speed
        target_indices = self.rng.choice(
            n_total, 
            size=num_needed, 
            p=probs, 
            replace=True # Preferential attachment allows multiple incoming
        )
        
        # --- Phase 3: Wiring ---
        for i, u in enumerate(nodes_seeking_new_connection):
            target_idx = target_indices[i]
            v = all_nodes[target_idx]
            
            # Validity Checks (O(1) usually)
            if u == v: 
                continue # Skip self-loops
            if self.G.has_edge(u, v): 
                continue # Skip existing edges
            
            # Add Connection
            self.G.add_edge(u, v)
            self._init_trust_one_way(u, v)
            self._init_trust_one_way(v, u)