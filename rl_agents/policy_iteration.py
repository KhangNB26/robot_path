# rl_agents/policy_iteration.py
"""
Simple policy iteration on small explored state-set starting from start state
"""
from collections import defaultdict

class PolicyIterationAgent:
    def __init__(self, mdp_model, gamma, max_iters):
        self.mdp = mdp_model
        self.gamma = gamma
        self.max_iters = max_iters
        self.V = {}
        self.pi = {}

    def _get_reachable_states(self, start_state, max_states=20000):
        """Get reachable states using list as queue"""
        visited = set()
        queue = [start_state]  # Using list instead of deque
        visited.add(start_state)
        queue_index = 0  # Track current position in queue
        
        while queue_index < len(queue) and len(visited) < max_states:
            s = queue[queue_index]
            queue_index += 1
            
            for a in self.mdp.actions:
                ns, _ = self.mdp.step(s, a)
                if ns not in visited:
                    visited.add(ns)
                    queue.append(ns)
        
        return visited

    def run(self, start_state):
        # Get reachable states
        visited = self._get_reachable_states(start_state)
        
        # Initialize policy and values
        for s in visited:
            self.pi[s] = self.mdp.actions[0]
            self.V[s] = 0.0

        # Policy iteration
        for it in range(self.max_iters):
            # Policy evaluation
            while True:
                delta = 0
                for s in visited:
                    if self.mdp.is_terminal(s[2]):
                        continue
                    
                    old_v = self.V[s]
                    a = self.pi[s]
                    ns, r = self.mdp.step(s, a)
                    self.V[s] = r + self.gamma * self.V.get(ns, 0.0)
                    delta = max(delta, abs(old_v - self.V[s]))
                
                if delta < 1e-3:
                    break

            # Policy improvement
            policy_stable = True
            for s in visited:
                if self.mdp.is_terminal(s[2]):
                    continue
                    
                old_a = self.pi[s]
                best_v = float('-inf')
                best_a = old_a
                
                for a in self.mdp.actions:
                    ns, r = self.mdp.step(s, a)
                    val = r + self.gamma * self.V.get(ns, 0.0)
                    if val > best_v:
                        best_v = val
                        best_a = a
                
                self.pi[s] = best_a
                if old_a != best_a:
                    policy_stable = False

            if policy_stable:
                break

        return self.pi, self.V
