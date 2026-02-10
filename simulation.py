import collections
import random
from config import SimConfig
from agent import AgentLogic
from environment import SocialNetwork

class Simulation:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.net = SocialNetwork(config)
        self.cascade_count = 0

    def run_single_cascade(self, cascade_id):
        self.cascade_count += 1
        is_fake = random.random() < self.cfg.global_fake_prob
        seed_node = random.choice(list(self.net.G.nodes()))
        
        queue = collections.deque([(seed_node, None)])
        visited = {seed_node}
        
        # Store tuple: (ActionType, MarginalReach)
        # This fixes the "Reach Bug" by storing exactly how many people 
        # were susceptible when the agent made the decision.
        actions_log = {} 
        
        n_shares = 0
        n_verifications = 0
        n_debunks = 0
        n_receivers = 0 
        total_payoff = 0.0

        while queue:
            current, sender = queue.popleft()
            n_receivers += 1 
            
            # --- Belief Formation ---
            if sender is None:
                reliability = 0.5
            else:
                # If rewiring happened, sender might no longer be a neighbor. Handle gracefully.
                if sender in self.net.G.nodes[current]['neighbor_trust']:
                    trust_params = self.net.G.nodes[current]['neighbor_trust'][sender]
                    reliability = AgentLogic.get_neighbor_reliability(trust_params['alpha'], trust_params['beta'])
                else:
                    reliability = 0.5 # Default if link broken recently

            global_fake_est = self.net.G.nodes[current]['fake_belief']
            p_true = AgentLogic.calculate_posterior_truth_prob(reliability, global_fake_est)
            
            # --- Marginal Reach Calculation ---
            # How many neighbors have NOT seen this item yet?
            reach = self.net.get_reach(current, visited)
            
            # Optimization: If reach is 0, rational agents usually Discard. 
            # But we run logic anyway for completeness.
            
            # --- Decision ---
            action = AgentLogic.decide(p_true, reach, self.cfg)
            
            # --- Execution ---
            
            if action == 'DISCARD':
                pass 
                
            elif action == 'SHARE':
                # Blind Share
                n_shares += 1
                outcome_type = 'shared_fake' if is_fake else 'shared_true'
                actions_log[current] = (outcome_type, reach)
                
                # Payoff Calculation (based on actual reality)
                if is_fake:
                    total_payoff -= (self.cfg.rep_loss * reach)
                else:
                    total_payoff += (self.cfg.rep_gain * reach)

                # Queue Neighbors
                for nbr in self.net.G.neighbors(current):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, current))
                        
            elif action == 'VERIFY':
                n_verifications += 1
                total_payoff -= self.cfg.verification_cost
                
                # --- NOISY DETECTION LOGIC ---
                detected_as_fake = False
                if is_fake:
                    # Roll for detection accuracy
                    if random.random() < self.cfg.detection_accuracy:
                        detected_as_fake = True
                    else:
                        detected_as_fake = False # FALSE NEGATIVE!
                else:
                    # Assuming no False Positives for simplicity (Truth is recognized as Truth)
                    detected_as_fake = False

                if detected_as_fake:
                    # Successful Debunk
                    n_debunks += 1
                    actions_log[current] = ('debunked', reach)
                    total_payoff += (self.cfg.rep_debunk * reach)
                    # Do NOT propagate
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

        # --- Post-Cascade Updates ---
        self._apply_updates(actions_log, is_fake)
        
        # --- Optional Rewiring ---
        if self.cfg.enable_rewiring and (self.cascade_count % self.cfg.rewiring_interval == 0):
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

    def _apply_updates(self, actions_log, item_is_fake):
        for agent, (action, marginal_reach) in actions_log.items():
            
            # 1. Update Reputation (Using Stored MARGINAL Reach)
            if action == 'shared_true':
                self.net.G.nodes[agent]['reputation'] += (self.cfg.rep_gain * marginal_reach)
            elif action == 'shared_fake':
                self.net.G.nodes[agent]['reputation'] -= (self.cfg.rep_loss * marginal_reach)
            elif action == 'debunked':
                self.net.G.nodes[agent]['reputation'] += (self.cfg.rep_debunk * marginal_reach)
            
            # 2. Update Neighbors' Trust
            # Note: Trust is updated for ALL current neighbors, even those who didn't see it,
            # representing "word gets out" about your behavior.
            for nbr in self.net.G.neighbors(agent):
                if action == 'shared_true':
                    self.net.update_trust(nbr, agent, True)
                elif action == 'shared_fake':
                    self.net.update_trust(nbr, agent, False)
                elif action == 'debunked':
                    self.net.update_trust(nbr, agent, True)
            
            # 3. Global Belief Update
            self.net.update_global_belief(agent, item_is_fake)