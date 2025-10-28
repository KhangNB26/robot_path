# rl_agents/value_iteration.py
"""
Tabular Value Iteration over a small MDP built from mdp/mdp_model.py
Note: This is a simplistic implementation for demonstration.
"""
import math
from collections import defaultdict

class ValueIterationAgent:
    def __init__(self, mdp_model, gamma, theta, max_iters):
        self.mdp = mdp_model
        self.gamma = gamma
        self.theta = theta
        self.max_iters = max_iters
        self.V = defaultdict(float)  # Initialize values to 0
        self.pi = {}

    def value_iteration(self):
        iteration = 0
        while iteration < self.max_iters:
            delta = 0
            # Loop through all states
            for state in self.mdp.get_all_states():
                if self.mdp.is_terminal(state[2]):
                    continue
                    
                # Store old value
                old_v = self.V[state]
                
                # Compute new value using Bellman equation
                max_v = float('-inf')
                for action in self.mdp.actions:
                    # Consider all possible next states and their probabilities
                    next_state, reward = self.mdp.step(state, action)
                    v = reward + self.gamma * self.V[next_state]
                    max_v = max(max_v, v)
                
                # Update value
                self.V[state] = max_v
                delta = max(delta, abs(old_v - self.V[state]))
            
            # Check convergence
            if delta < self.theta:
                break
                
            iteration += 1

    def extract_policy(self):
        # Extract optimal policy from value function
        for state in self.mdp.get_all_states():
            if self.mdp.is_terminal(state[2]):
                self.pi[state] = (0,0)
                continue
                
            # Find action that maximizes value
            max_v = float('-inf')
            best_action = None
            
            for action in self.mdp.actions:
                next_state, reward = self.mdp.step(state, action)
                v = reward + self.gamma * self.V[next_state]
                if v > max_v:
                    max_v = v
                    best_action = action
                    
            self.pi[state] = best_action

    def run(self, start_state):
        self.value_iteration()
        self.extract_policy()
        return self.pi, self.V
