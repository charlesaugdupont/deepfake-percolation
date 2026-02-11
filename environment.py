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
            seed=int(rng.integers(0, 1_000_000))
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
        Dynamic Topology:
        1. Identify weak links (trust < threshold).
        2. Remove them.
        3. Create new random links to maintain density.
        """
        edges_to_remove = []
        nodes_seeking_partners = []

        # 1. Identify Cuts
        for u in self.G.nodes():
            neighbors = list(self.G.neighbors(u))
            for v in neighbors:
                # Undirected graph: check u's trust in v
                trust_data = self.G.nodes[u]['neighbor_trust'].get(v)
                if not trust_data: continue

                trust_score = AgentLogic.get_neighbor_reliability(trust_data['alpha'], trust_data['beta'])
                
                if trust_score < self.cfg.rewire_trust_threshold:
                    # Mark for removal. (We'll handle the undirected duplicate edge naturally)
                    if self.G.has_edge(u, v):
                        edges_to_remove.append((u, v))
                        nodes_seeking_partners.append(u)

        # 2. Execute Cuts
        unique_cuts = set(tuple(sorted(e)) for e in edges_to_remove)
        for u, v in unique_cuts:
            if self.G.has_edge(u, v):
                self.G.remove_edge(u, v)
                # Cleanup trust memory
                if v in self.G.nodes[u]['neighbor_trust']:
                    del self.G.nodes[u]['neighbor_trust'][v]
                if u in self.G.nodes[v]['neighbor_trust']:
                    del self.G.nodes[v]['neighbor_trust'][u]

        # 3. Form New Ties (Random Rewiring)
        # Each node that lost a link tries to find a new random partner
        all_nodes = list(self.G.nodes())
        for u in nodes_seeking_partners:
            # Try 5 times to find a non-neighbor
            for _ in range(5):
                candidate = self.rng.choice(all_nodes)
                if candidate != u and not self.G.has_edge(u, candidate):
                    self.G.add_edge(u, candidate)
                    # Initialize trust BOTH ways
                    self.init_edge_trust(u, candidate)
                    self.init_edge_trust(candidate, u)
                    break