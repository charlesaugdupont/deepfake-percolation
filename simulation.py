import collections
import random
from config import SimConfig
from agent import AgentLogic
from environment import SocialNetwork

class Simulation:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.net = SocialNetwork(config)

    def run_single_cascade(self, cascade_id):
        is_fake = random.random() < self.cfg.global_fake_prob
        seed_node = random.choice(list(self.net.G.nodes()))
        
        queue = collections.deque([(seed_node, None)])
        visited = {seed_node}
        
        actions_taken = {} 
        
        # Stats
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
                trust_params = self.net.G.nodes[current]['neighbor_trust'][sender]
                reliability = AgentLogic.get_neighbor_reliability(
                    trust_params['alpha'], trust_params['beta']
                )

            global_fake_est = self.net.G.nodes[current]['fake_belief']
            p_true = AgentLogic.calculate_posterior_truth_prob(reliability, global_fake_est)
            
            # --- Estimate Reach ---
            reach = self.net.get_reach(current, visited)
            
            # --- Decision ---
            action = AgentLogic.decide(p_true, reach, self.cfg)
            
            # --- Execution ---
            if action == 'DISCARD':
                pass 
                
            elif action == 'SHARE':
                n_shares += 1
                if is_fake:
                    actions_taken[current] = 'shared_fake'
                    total_payoff -= (self.cfg.rep_loss * reach)
                else:
                    actions_taken[current] = 'shared_true'
                    total_payoff += (self.cfg.rep_gain * reach)

                for nbr in self.net.G.neighbors(current):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, current))
                        
            elif action == 'VERIFY':
                n_verifications += 1
                total_payoff -= self.cfg.verification_cost
                
                if is_fake:
                    n_debunks += 1
                    actions_taken[current] = 'debunked'
                    total_payoff += (self.cfg.rep_debunk * reach)
                else:
                    n_shares += 1
                    actions_taken[current] = 'shared_true'
                    total_payoff += (self.cfg.rep_gain * reach)
                    
                    for nbr in self.net.G.neighbors(current):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append((nbr, current))

        # --- Post-Cascade Updates ---
        self._apply_updates(actions_taken, is_fake)
        
        return {
            'is_fake': is_fake,
            'size': len(visited),
            'shares': n_shares,
            'verifications': n_verifications,
            'receivers': n_receivers,
            'debunks': n_debunks,
            'utility': total_payoff
        }

    def _apply_updates(self, actions, item_is_fake):
        for agent, action in actions.items():
            reach = self.net.get_reach(agent, set())
            
            if action == 'shared_true':
                self.net.update_reputation(agent, self.cfg.rep_gain * reach)
            elif action == 'shared_fake':
                self.net.update_reputation(agent, -self.cfg.rep_loss * reach)
            elif action == 'debunked':
                self.net.update_reputation(agent, self.cfg.rep_debunk * reach)
            
            for nbr in self.net.G.neighbors(agent):
                if action == 'shared_true':
                    self.net.update_trust(nbr, agent, True)
                elif action == 'shared_fake':
                    self.net.update_trust(nbr, agent, False)
                elif action == 'debunked':
                    self.net.update_trust(nbr, agent, True)
            
            self.net.update_global_belief(agent, item_is_fake)