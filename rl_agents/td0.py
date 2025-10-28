# rl_agents/td0.py
"""
TD(0) learning for state-value function (on-policy) using random policy or given policy
"""
import random
from collections import defaultdict

class TD0Agent:
    def __init__(self, mdp_model, alpha, gamma, episodes, max_steps, policy=None):
        self.mdp = mdp_model
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.max_steps = max_steps
        # policy: function mapping state -> action
        self.policy = policy or (lambda s: random.choice(self.mdp.actions))
        self.V = defaultdict(float)
        self.policy_dict = {}  # Add policy dictionary

    def run(self, start_state):
        # Run TD(0) learning
        for ep in range(self.episodes):
            state = start_state
            for t in range(self.max_steps):
                a = self.policy(state)
                ns, r = self.mdp.step(state, a)
                self.V[state] += self.alpha * (r + self.gamma * self.V[ns] - self.V[state])
                self.policy_dict[state] = a  # Store action in policy dictionary
                state = ns
                if self.mdp.is_terminal(state[2]):
                    break
        
        return self.policy_dict, self.V
