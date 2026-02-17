import collections
import numpy as np

from config import SimConfig
from agent import AgentLogic
from environment import SocialNetwork

class Simulation:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.random_seed)
        self.net = SocialNetwork(config, self.rng)
        self.cascade_count = 0
        self.last_cascade_edges = []
        self.all_nodes = list(self.net.G.nodes())

    def run_single_cascade(self):
        self.cascade_count += 1
        is_fake = self.rng.random() < self.cfg.global_fake_prob
        seed_node = self.rng.choice(self.all_nodes)
        
        queue = collections.deque([(seed_node, None)])
        visited = {seed_node}
        parent_map = {}
        
        # Store tuple: (ActionType, MarginalReach)
        actions_log = {} 
        
        n_shares = 0
        n_verifications = 0
        n_debunks = 0
        n_receivers = 0 
        total_payoff = 0.0
        cascade_edges = []

        while queue:
            current, sender = queue.popleft()
            n_receivers += 1
            
            # belief formation 
            if sender is None:
                reliability = 0.5
            else:
                # If rewiring happened, sender might no longer be a neighbor. Handle gracefully.
                if sender in self.net.G.nodes[current]['neighbor_trust']:
                    trust_params = self.net.G.nodes[current]['neighbor_trust'][sender]
                    reliability = AgentLogic.get_neighbor_reliability(trust_params['alpha'], trust_params['beta'])
                else:
                    reliability = AgentLogic.get_neighbor_reliability(self.cfg.neighbor_truth_prior_alpha, self.cfg.neighbor_false_prior_beta)

            stats = self.net.G.nodes[current]['true_belief']
            global_true_est = stats['alpha'] / (stats['alpha'] + stats['beta'])
            p_true = AgentLogic.calculate_posterior_truth_prob(reliability, global_true_est)

            # agents that have not seen the item
            reach = self.net.get_reach(current, visited)
            
            # decision
            action = AgentLogic.decide(p_true, reach, self.cfg)
            
            # execution
            if action == 'DISCARD':
                pass
            
            elif action == 'SHARE':
                n_shares += 1
                outcome_type = 'shared_fake' if is_fake else 'shared_true'
                actions_log[current] = (outcome_type, reach)
                
                # calculate payoff
                if is_fake:
                    total_payoff -= (self.cfg.rep_loss * reach)
                else:
                    total_payoff += (self.cfg.rep_gain * reach)

                # queue neighbors
                for nbr in self.net.G.neighbors(current):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, current))
                        parent_map[nbr] = current
                        cascade_edges.append((current, nbr))
                        
            elif action == 'VERIFY':
                n_verifications += 1
                total_payoff -= self.cfg.verification_cost
                
                # noisy detection
                detected_as_fake = False
                if is_fake:
                    # roll for detection accuracy
                    if self.rng.random() < self.cfg.verification_accuracy:
                        detected_as_fake = True
                    else:
                        detected_as_fake = False # false negative
                else:
                    # assuming no False Positives for simplicity
                    detected_as_fake = False

                if detected_as_fake:
                    n_debunks += 1
                else:
                    # Shared (either correctly or via False Negative)
                    n_shares += 1
                    outcome_type = 'shared_fake' if is_fake else 'shared_true'
                    actions_log[current] = (outcome_type, reach)
                    
                    if is_fake:
                        # False Negative Penalty
                        total_payoff -= (self.cfg.rep_loss * reach)
                    else:
                        total_payoff += (self.cfg.rep_gain * reach)
                    
                    # Propagate
                    for nbr in self.net.G.neighbors(current):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append((nbr, current))
                            parent_map[nbr] = current
                            cascade_edges.append((current, nbr))

        # Post-Cascade Updates 
        self._apply_updates(actions_log, visited, parent_map, is_fake)

        self.last_cascade_edges = cascade_edges

        # stochastic rewiring after each cascasde
        if self.cfg.enable_rewiring and self.cascade_count % self.cfg.rewiring_interval == 0:
            self.net.rewire_network()

        return {
            'is_fake': is_fake,
            'size': len(visited),
            'shares': n_shares,
            'verifications': n_verifications,
            'receivers': n_receivers,
            'debunks': n_debunks,
            'utility': total_payoff
        }

    def _apply_updates(self, actions_log, visited, parent_map, item_is_fake):
        # reputation updates (only agents who acted)
        for agent, (action, marginal_reach) in actions_log.items():
            if action == 'shared_true':
                self.net.G.nodes[agent]['reputation'] += (self.cfg.rep_gain * marginal_reach)
            elif action == 'shared_fake':
                self.net.G.nodes[agent]['reputation'] -= (self.cfg.rep_loss * marginal_reach)

        # global belief updates (ALL who saw item)
        for agent in visited:
            self.net.update_global_belief(agent, item_is_fake)

        # trust updates (receiver -> actual sender only)
        for receiver, sender in parent_map.items():
            if item_is_fake:
                self.net.update_trust(receiver, sender, False)
            else:
                self.net.update_trust(receiver, sender, True)
