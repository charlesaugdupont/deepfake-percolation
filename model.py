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
        
    def _build_network(self) -> nx.Graph:
        """Generates the social network topology."""
        if self.cfg.network_type == 'watts_strogatz':
            return nx.watts_strogatz_graph(
                n=self.cfg.num_agents, 
                k=self.cfg.ws_k, 
                p=self.cfg.ws_p
            )
        elif self.cfg.network_type == 'barabasi_albert':
            return nx.barabasi_albert_graph(
                n=self.cfg.num_agents, 
                m=self.cfg.ba_m
            )
        else:
            raise ValueError(f"Unknown network type: {self.cfg.network_type}")

    def _initialize_agent_states(self):
        """
        Sets up initial trust priors for edges.
        SIMPLIFIED: All agents have same fixed w (no variance).
        """
        # 1. Node Initialization - SIMPLIFIED
        for node in self.G.nodes():
            self.G.nodes[node]['utility'] = 0.0
            
        # 2. Edge Initialization (Trust Priors)
        alpha_init = self.cfg.initial_trust_prior * self.cfg.baseline_trust
        beta_init  = self.cfg.initial_trust_prior * (1.0 - self.cfg.baseline_trust)
        
        for u, v in self.G.edges():
            self.G[u][v]['trust'] = {
                u: {'alpha': alpha_init, 'beta': beta_init},
                v: {'alpha': alpha_init, 'beta': beta_init}
            }
            
    def get_trust_scores(self, observer: int, target: int):
        """Helper to get alpha/beta that 'observer' has for 'target'."""
        return self.G[observer][target]['trust'][observer]

    def update_trust(self, observer: int, target: int, evidence_is_fake: bool):
        """Bayesian update of trust scores based on verification result."""
        scores = self.G[observer][target]['trust'][observer]
        if evidence_is_fake:
            scores['beta'] += 1
        else:
            scores['alpha'] += 1

    def apply_payoff(self, agent_id: int, action: str, is_ground_truth_fake: bool, 
                     detected_as_fake: bool, item_cost: float):
        """Updates the cumulative utility of an agent based on the OUTCOME of their action."""
        payoff = 0.0
        cfg = self.cfg
        
        # Compute actual payoff values
        L = cfg.R * cfg.loss_aversion_ratio
        D = cfg.R * cfg.debunking_premium_ratio
        
        if action == 'DISCARD':
            payoff = 0.0
            
        elif action == 'VERIFY':
            payoff -= item_cost
            
            if detected_as_fake:
                if is_ground_truth_fake:
                    payoff += D  # Successful Debunk
                else:
                    payoff += 0  # False Positive
            else:
                if is_ground_truth_fake:
                    payoff -= L  # False Negative
                else:
                    payoff += cfg.R  # True Negative
                    
        elif action == 'SHARE':
            if is_ground_truth_fake:
                payoff -= L
            else:
                payoff += cfg.R
                
        self.G.nodes[agent_id]['utility'] += payoff

    def run_cascade(self, source_node: int, is_ground_truth_fake: bool):
        """
        Simulates a single piece of news spreading from source_node.
        Returns statistics about the cascade.
        """
        queue = collections.deque()
        received_set = set()
        
        # Initial Injection
        received_set.add(source_node)
        for neighbor in self.G.neighbors(source_node):
            queue.append((neighbor, source_node))
            received_set.add(neighbor)
            
        stats = {
            'total_reached': 1,
            'total_shared': 0,
            'total_verified': 0,
            'total_discarded': 0,
        }
        
        while queue:
            current, sender = queue.popleft()
            stats['total_reached'] += 1
            
            # 1. Get trust in sender
            scores = self.get_trust_scores(current, sender)
            
            # 2. Form belief - SIMPLIFIED: use fixed w from config
            p_fake = AgentLogic.estimate_p_fake(
                scores['alpha'], 
                scores['beta'], 
                self.cfg.w,  # Fixed for all agents
                self.cfg
            )

            # 3. Fixed cost
            item_cost = self.cfg.C
            
            # 4. Make Decision
            action = AgentLogic.make_cpt_decision(p_fake, item_cost, self.cfg)
            final_action_share = False
            detected_as_fake = False
            
            if action == 'VERIFY':
                stats['total_verified'] += 1
                detected_as_fake = AgentLogic.perform_verification(is_ground_truth_fake, self.cfg)
                self.update_trust(current, sender, detected_as_fake)
                
                if detected_as_fake:
                    final_action_share = False
                    stats['total_discarded'] += 1
                else:
                    final_action_share = True
            
            elif action == 'SHARE':
                final_action_share = True
                
            elif action == 'DISCARD':
                final_action_share = False
                stats['total_discarded'] += 1

            # 5. Apply payoff
            self.apply_payoff(current, action, is_ground_truth_fake, detected_as_fake, item_cost)
            
            # 6. Propagation
            if final_action_share:
                stats['total_shared'] += 1
                for neighbor in self.G.neighbors(current):
                    if neighbor not in received_set:
                        received_set.add(neighbor)
                        queue.append((neighbor, current))
                        
        return stats