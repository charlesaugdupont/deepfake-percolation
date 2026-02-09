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
        Sets up initial w (uncertainty weights) for nodes
        and alpha/beta (trust priors) for edges.
        """
        # 1. Node Initialization (w)
        for node in self.G.nodes():
            # Draw w from a bounded normal or beta distribution
            w = np.random.normal(self.cfg.w_mean, self.cfg.w_std)
            w = np.clip(w, 0.0, 1.0)
            self.G.nodes[node]['w'] = w
            self.G.nodes[node]['reputation'] = 0.0
            
        # 2. Edge Initialization (Trust Priors)
        # alpha = verified TRUE count, beta = verified FAKE count
        alpha_init = self.cfg.initial_trust_prior * self.cfg.baseline_trust
        beta_init  = self.cfg.initial_trust_prior * (1.0 - self.cfg.baseline_trust)
        
        for u, v in self.G.edges():
            # NetworkX edges are undirected, but trust is directed.
            # We must store attributes for both directions: u->v and v->u.
            # We'll use a dictionary on the edge to store 'trust_data'.
            # key: source_node_id, value: {'alpha': ..., 'beta': ...}
            
            self.G[u][v]['trust'] = {
                u: {'alpha': alpha_init, 'beta': beta_init}, # Trust u holds for v? No, usually localized.
                v: {'alpha': alpha_init, 'beta': beta_init}  # Trust v holds for u?
            }
            # Correction: It's cleaner to think: 
            # If I am 'u', I have an alpha/beta count for 'v'.
            
    def get_trust_scores(self, observer: int, target: int):
        """Helper to get alpha/beta that 'observer' has for 'target'."""
        # In undirected graph G[u][v], we stored a dict keyed by the *observer*
        return self.G[observer][target]['trust'][observer]

    def update_trust(self, observer: int, target: int, evidence_is_fake: bool):
        """Bayesian update of trust scores based on verification result."""
        scores = self.G[observer][target]['trust'][observer]
        if evidence_is_fake:
            scores['beta'] += 1
        else:
            scores['alpha'] += 1

    def run_cascade(self, source_node: int, is_ground_truth_fake: bool):
        """
        Simulates a single piece of news spreading from source_node.
        Returns statistics about the cascade.
        """
        # Queue for propagation: (current_node, sender_node)
        queue = collections.deque()
        
        # Track who has seen this specific item to prevent cycles
        received_set = set()
        
        # Initial Injection
        received_set.add(source_node)
        # Source sends to all neighbors
        for neighbor in self.G.neighbors(source_node):
            queue.append((neighbor, source_node))
            received_set.add(neighbor)
            
        stats = {
            'total_reached': 1, # Source
            'total_shared': 0,
            'total_verified': 0,
            'total_discarded': 0,
            'detected_fakes': 0, # Successful debunks
            'false_positives': 0, # True news labeled fake
            'false_negatives': 0, # Fake news labeled true
        }
        
        while queue:
            current, sender = queue.popleft()
            stats['total_reached'] += 1
            
            # 1. Retrieve State
            scores = self.get_trust_scores(current, sender)
            w = self.G.nodes[current]['w']
            
            # 2. Form Belief
            p_fake = AgentLogic.estimate_p_fake(scores['alpha'], scores['beta'], w, self.cfg)
            
            # 3. Make Decision
            action = AgentLogic.make_decision(p_fake, self.cfg)
            
            final_action_share = False
            
            if action == 'DISCARD':
                stats['total_discarded'] += 1
                
            elif action == 'SHARE':
                # Sharing without verification (Gambling)
                final_action_share = True
                
            elif action == 'VERIFY':
                stats['total_verified'] += 1
                
                # Perform Noisy Verification
                detected_as_fake = AgentLogic.perform_verification(is_ground_truth_fake, self.cfg)
                
                # Update Trust based on DETECTION (not necessarily ground truth)
                self.update_trust(current, sender, detected_as_fake)
                
                if detected_as_fake:
                    # Agent thinks it's fake -> Debunk & Discard
                    if is_ground_truth_fake:
                        stats['detected_fakes'] += 1
                    else:
                        stats['false_positives'] += 1 # Oops, debunked real news
                    
                    # Do not share
                    final_action_share = False 
                    
                else:
                    # Agent thinks it's true -> Share
                    if is_ground_truth_fake:
                        stats['false_negatives'] += 1 # Oops, verifying let a fake through
                    
                    final_action_share = True
                    # Update trust (Positive reinforcement for sender)
                    # self.update_trust(current, sender, False) -> Already handled above
            
            # 4. Propagation
            if final_action_share:
                stats['total_shared'] += 1
                for neighbor in self.G.neighbors(current):
                    if neighbor not in received_set:
                        received_set.add(neighbor)
                        queue.append((neighbor, current))
                        
        return stats