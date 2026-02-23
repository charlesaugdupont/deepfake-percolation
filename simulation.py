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
        self.all_nodes = list(self.net.G.nodes())

    def run_single_cascade(self):
        self.cascade_count += 1
        item_is_fake = self.rng.random() < self.cfg.global_fake_prob
        seed_node = self.rng.choice(self.all_nodes)
        
        queue = collections.deque([(seed_node, None)])
        visited = {seed_node}
        parent_map = {}
        
        # Store tuple: (ActionType, MarginalReach)
        actions_log = {}

        verification_log = []
        
        n_shares = 0
        n_verifications = 0
        n_receivers = 0
        total_payoff = 0.0

        actual_tpr = self.cfg.verification_tpr
        actual_fpr = self.cfg.verification_fpr

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
                    reliability = AgentLogic.compute_beta_mean(trust_params['alpha'], trust_params['beta'])
                else:
                    reliability = AgentLogic.compute_beta_mean(self.cfg.neighbor_truth_prior_alpha, self.cfg.neighbor_false_prior_beta)

            # global belief
            node_stats = self.net.G.nodes[current]
            true_belief = node_stats["true_belief"]
            global_true_est = true_belief['alpha'] / (true_belief['alpha'] + true_belief['beta'])
            p_true = AgentLogic.calculate_posterior_truth_prob(reliability, global_true_est)

            # verification tech belief
            tpr_belief = node_stats["tech_tpr_belief"]
            fpr_belief = node_stats["tech_fpr_belief"]
            est_tpr = AgentLogic.compute_beta_mean(tpr_belief["alpha"], tpr_belief["beta"])
            est_fpr = AgentLogic.compute_beta_mean(fpr_belief["alpha"], fpr_belief["beta"])

            # estimate potential reach
            reach = self.net.G.degree[current]
            
            # decision
            action = AgentLogic.decide(p_true, reach, est_tpr, est_fpr, self.cfg)
            
            # execution
            if action == 'DISCARD':
                pass
            
            elif action == 'SHARE':
                n_shares += 1
                outcome_type = 'shared_fake' if item_is_fake else 'shared_true'
                actions_log[current] = (outcome_type, reach)
                
                # calculate payoff
                if item_is_fake:
                    total_payoff -= (self.cfg.rep_loss * reach)
                else:
                    total_payoff += (self.cfg.rep_gain * reach)

                # queue neighbors
                for nbr in self.net.G.neighbors(current):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append((nbr, current))
                        parent_map[nbr] = current
                        
            elif action == 'VERIFY':
                n_verifications += 1
                total_payoff -= self.cfg.verification_cost

                tool_says_fake = False

                if item_is_fake:
                    if self.rng.random() < actual_tpr:
                        tool_says_fake = True   # True Positive
                    else:
                        tool_says_fake = False  # False Negative
                else:
                    if self.rng.random() < actual_fpr:
                        tool_says_fake = True # False Positive
                    else:
                        tool_says_fake = False # True Negative

                verification_log.append((current, tool_says_fake))

                if tool_says_fake:
                    # Agent believes the tool -> Discard
                    # No reputation change; propagation stops here.
                    pass
                else:
                    # Agent believes the tool (it said True) -> Share
                    n_shares += 1
                    outcome_type = 'shared_fake' if item_is_fake else 'shared_true'
                    actions_log[current] = (outcome_type, reach)
                    if item_is_fake:
                        total_payoff -= (self.cfg.rep_loss * reach)
                    else:
                        total_payoff += (self.cfg.rep_gain * reach)
                    # Propagate
                    for nbr in self.net.G.neighbors(current):
                        if nbr not in visited:
                            visited.add(nbr)
                            queue.append((nbr, current))
                            parent_map[nbr] = current

        # Post-Cascade Updates 
        self._apply_updates(actions_log, verification_log, visited, parent_map, item_is_fake)

        # stochastic rewiring after each cascasde
        if self.cfg.enable_rewiring and self.cascade_count % self.cfg.rewiring_interval == 0:
            self.net.rewire_network()

        return {
            'is_fake': item_is_fake,
            'size': len(visited),
            'shares': n_shares,
            'verifications': n_verifications,
            'receivers': n_receivers,
            'utility': total_payoff
        }

    def _apply_updates(self, actions_log, verification_log, visited, parent_map, item_is_fake):
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

        # verification tech belief updates
        for agent_id, tool_said_fake in verification_log:
            tpr_stats = self.net.G.nodes[agent_id]['tech_tpr_belief']
            fpr_stats = self.net.G.nodes[agent_id]['tech_fpr_belief']
            if item_is_fake:
                if tool_said_fake:
                    tpr_stats["alpha"] += 1
                else:
                    tpr_stats["beta"] += 1
            else:
                if tool_said_fake:
                    fpr_stats["alpha"] += 1
                else:
                    fpr_stats["beta"] += 1