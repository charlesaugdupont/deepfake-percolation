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
            self.G.nodes[node]['true_belief'] = {
                'alpha': self.cfg.global_truth_prior_alpha,
                'beta': self.cfg.global_false_prior_beta
            }
            self.G.nodes[node]['neighbor_trust'] = {}
            
            # Initialize trust for existing edges
            for nbr in self.G.neighbors(node):
                self.init_edge_trust(node, nbr)

    def init_edge_trust(self, node, nbr):
        """Helper to init a single edge"""
        self.G.nodes[node]['neighbor_trust'][nbr] = {
            'alpha': self.cfg.neighbor_truth_prior_alpha, 
            'beta': self.cfg.neighbor_false_prior_beta
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
            return
            
        stats = self.G.nodes[observer]['neighbor_trust'][sender]
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
        Rewire edges whose average reliability falls below threshold.
        Each removed undirected edge produces exactly one reconnection
        attempt per endpoint.
        """
        edges_to_remove = []

        # --------------------------------------------------
        # 1️⃣ Identify weak edges (edges() already unique)
        # --------------------------------------------------
        for u, v in self.G.edges():

            trust_uv = self.G.nodes[u]["neighbor_trust"].get(v)
            trust_vu = self.G.nodes[v]["neighbor_trust"].get(u)

            # Safety check (edge might be partially cleaned)
            if not trust_uv or not trust_vu:
                continue

            rel_uv = AgentLogic.get_neighbor_reliability(
                trust_uv["alpha"], trust_uv["beta"]
            )

            rel_vu = AgentLogic.get_neighbor_reliability(
                trust_vu["alpha"], trust_vu["beta"]
            )

            if rel_uv < self.cfg.rewire_trust_threshold or rel_vu < self.cfg.rewire_trust_threshold:
                edges_to_remove.append((u, v))

        # --------------------------------------------------
        # 2️⃣ Remove weak edges + clean trust memory
        # --------------------------------------------------
        nodes_seeking = set()

        for u, v in edges_to_remove:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)

            # Clean trust memory both directions
            self.G.nodes[u]["neighbor_trust"].pop(v, None)
            self.G.nodes[v]["neighbor_trust"].pop(u, None)

            nodes_seeking.add(u)
            nodes_seeking.add(v)

        # --------------------------------------------------
        # 3️⃣ Each endpoint seeks ONE new partner
        # --------------------------------------------------
        all_nodes = list(self.G.nodes())

        for u in nodes_seeking:

            # Candidates: not self, not already connected
            potential = [
                v for v in all_nodes
                if v != u and not self.G.has_edge(u, v)
            ]

            if not potential:
                continue

            v_new = self.rng.choice(potential)

            self.G.add_edge(u, v_new)

            # Initialize trust BOTH ways using existing helper
            self.init_edge_trust(u, v_new)
            self.init_edge_trust(v_new, u)

