import networkx as nx
from config import SimConfig
from agent import AgentLogic

class SocialNetwork:
    def __init__(self, config: SimConfig, rng):
        self.cfg = config
        self.rng = rng

        self.G = nx.barabasi_albert_graph(
            n=config.num_agents,
            m=config.num_connections,
            seed=self.rng
        )
        self.initialize_state()

    def initialize_state(self):
        for node in self.G.nodes():
            self.G.nodes[node]["reputation"] = 0.0
            self.G.nodes[node]['fake_belief'] = self.cfg.initial_fake_belief
            self.G.nodes[node]['neighbor_trust'] = {}
            
            # Initialize trust for existing edges
            for nbr in self.G.neighbors(node):
                self.init_edge_trust(node, nbr)

    def init_edge_trust(self, node, nbr):
        """Helper to init a single edge"""
        self.G.nodes[node]['neighbor_trust'][nbr] = {
            'alpha': self.cfg.prior_alpha, 
            'beta': self.cfg.prior_beta
        }

    def get_reach(self, node_id, visited_set):
        """
        Marginal Reach: Count of neighbors who are NOT in visited_set.
        """
        neighbors = list(self.G.neighbors(node_id))
        unseen = [n for n in neighbors if n not in visited_set]
        return len(unseen)

    def update_trust(self, observer, sender, sender_was_truthful: bool):
        if sender not in self.G.nodes[observer]['neighbor_trust']:
            return # Edge might have been removed
            
        stats = self.G.nodes[observer]['neighbor_trust'][sender]
        if sender_was_truthful:
            stats['alpha'] += 1
        else:
            stats['beta'] += 1
            
    def update_global_belief(self, agent_id, item_was_fake: bool):
        lr = 0.1
        current = self.G.nodes[agent_id]['fake_belief']
        observed = 1.0 if item_was_fake else 0.0
        self.G.nodes[agent_id]['fake_belief'] = (1 - lr) * current + lr * observed

    def rewire_network(self):
        """
        Rewire edges whose trust falls below the threshold.
        Each removed undirected edge produces exactly one reconnection
        attempt per endpoint. Prevents duplicate removals and degree bias.
        """
        edges_to_remove = []

        # --------------------------------------------------
        # 1️⃣ Identify weak edges (avoid double counting)
        # --------------------------------------------------
        for u, v in self.G.edges():
            trust_uv = self.trust[u][v]
            trust_vu = self.trust[v][u]

            # You can choose asymmetric or average trust rule
            avg_trust = 0.5 * (trust_uv + trust_vu)

            if avg_trust < self.config.rewire_trust_threshold:
                edges_to_remove.append((u, v))

        # Deduplicate just in case (edges() should already be unique)
        unique_cuts = set(tuple(sorted(edge)) for edge in edges_to_remove)

        # --------------------------------------------------
        # 2️⃣ Remove weak edges
        # --------------------------------------------------
        for u, v in unique_cuts:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)

        # --------------------------------------------------
        # 3️⃣ Each endpoint seeks exactly one new partner
        # --------------------------------------------------
        nodes_seeking_partners = {node for edge in unique_cuts for node in edge}

        for u in nodes_seeking_partners:

            # Potential candidates: not self, not already connected
            potential = [
                v for v in self.G.nodes()
                if v != u and not self.G.has_edge(u, v)
            ]

            if not potential:
                continue

            # Random new partner (uses centralized RNG)
            v_new = self.rng.choice(potential)

            self.G.add_edge(u, v_new)

            # Initialize trust symmetrically
            self.trust[u][v_new] = self.config.prior_alpha
            self.trust[v_new][u] = self.config.prior_alpha
