# rl_agents/td_lambda.py
"""
TD(lambda) for state-values using eligibility traces (accumulating)
"""
from collections import defaultdict
import random

class TDLambdaAgent:
    def __init__(self, mdp_model, alpha, gamma, lam, episodes, max_steps, policy=None):
        self.mdp = mdp_model
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.episodes = episodes
        self.max_steps = max_steps
        self.policy = policy or (lambda s: random.choice(self.mdp.actions))
        self.V = defaultdict(float)

    def run(self, start_state):
        for ep in range(self.episodes):
            # eligibility traces
            E = defaultdict(float)
            state = start_state
            for t in range(self.max_steps):
                a = self.policy(state)
                ns, r = self.mdp.step(state, a)
                delta = r + self.gamma * self.V[ns] - self.V[state]
                E[state] += 1.0
                for s in list(E.keys()):
                    self.V[s] += self.alpha * delta * E[s]
                    E[s] = self.gamma * self.lam * E[s]
                state = ns
                if self.mdp.is_terminal(state[2]):
                    break
        return self.policy, self.V
