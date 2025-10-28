# rl_agents/q_learning.py
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, mdp_model, alpha, gamma, epsilon, episodes, max_steps):
        self.mdp = mdp_model
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.max_steps = max_steps
        self.Q = defaultdict(lambda: {a:0.0 for a in self.mdp.actions})

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            qvals = self.Q[state]
            return max(qvals.items(), key=lambda kv: kv[1])[0]

    def run(self, start_state):
        for ep in range(self.episodes):
            state = start_state
            for t in range(self.max_steps):
                a = self.choose_action(state)
                ns, r = self.mdp.step(state, a)
                best_next = max(self.Q[ns].values()) if ns in self.Q else 0.0
                self.Q[state][a] += self.alpha * (r + self.gamma * best_next - self.Q[state][a])
                state = ns
                if self.mdp.is_terminal(state[2]):
                    break
        # derive policy
        pi = {}
        for s, actions in self.Q.items():
            best = max(actions.items(), key=lambda kv: kv[1])[0]
            pi[s] = best
        return pi, self.Q
