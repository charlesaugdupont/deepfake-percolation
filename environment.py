import networkx as nx
from config import SimConfig

class SocialNetwork:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.G = nx.barabasi_albert_graph(n=config.num_agents, m=config.num_connections)
        self.initialize_state()

    def initialize_state(self):
        for node in self.G.nodes():
            self.G.nodes[node]['reputation'] = 0.0
            self.G.nodes[node]['fake_belief'] = self.cfg.initial_fake_belief
            
            # Initialize edge beliefs (Directed logic on undirected graph)
            # q_ij: Node i's belief about Node j
            self.G.nodes[node]['neighbor_trust'] = {}
            for nbr in self.G.neighbors(node):
                self.G.nodes[node]['neighbor_trust'][nbr] = {
                    'alpha': self.cfg.prior_alpha, 
                    'beta': self.cfg.prior_beta
                }

    def get_reach(self, node_id, visited_set):
        """
        Estimate downstream reach. 
        Minimal model: Simply the number of unvisited neighbors.
        """
        neighbors = list(self.G.neighbors(node_id))
        unseen = [n for n in neighbors if n not in visited_set]
        return len(unseen)

    def update_reputation(self, agent_id, delta):
        self.G.nodes[agent_id]['reputation'] += delta

    def update_trust(self, observer, sender, sender_was_truthful: bool):
        """Update Beta distribution for q_ij"""
        stats = self.G.nodes[observer]['neighbor_trust'][sender]
        if sender_was_truthful:
            stats['alpha'] += 1
        else:
            stats['beta'] += 1
            
    def update_global_belief(self, agent_id, item_was_fake: bool):
        """Simple exponential moving average or count-based update"""
        # Using a small learning rate for stability
        lr = 0.1
        current = self.G.nodes[agent_id]['fake_belief']
        observed = 1.0 if item_was_fake else 0.0
        self.G.nodes[agent_id]['fake_belief'] = (1 - lr) * current + lr * observed