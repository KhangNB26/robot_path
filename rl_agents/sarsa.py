# rl_agents/sarsa.py
import random
import numpy as np
from collections import defaultdict

class SarsaAgent:
    def __init__(self, mdp_model, alpha, gamma, epsilon, episodes, max_steps):
        self.mdp = mdp_model
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.episodes = episodes
        self.max_steps = max_steps
        # Initial Q-values with optimistic initialization
        self.Q = defaultdict(lambda: {a: 1.0 for a in self.mdp.actions})
        self.episode_rewards = []  # Track rewards per episode
        
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.choice(self.mdp.actions)
        else:
            # Break ties randomly for better exploration
            qvals = self.Q[state]
            max_q = max(qvals.values())
            best_actions = [a for a, q in qvals.items() if q == max_q]
            return random.choice(best_actions)

    def run(self, start_state):
        """Run SARSA algorithm"""
        print(f"Starting SARSA training for {self.episodes} episodes...")
        
        for ep in range(self.episodes):
            state = start_state
            action = self.choose_action(state)  # Choose A from S using policy
            episode_reward = 0
            
            for t in range(self.max_steps):
                # Take action, observe R, S'
                next_state, reward = self.mdp.step(state, action)
                episode_reward += reward
                
                if self.mdp.is_terminal(next_state[2]):
                    # Terminal state update
                    self.Q[state][action] += self.alpha * (reward - self.Q[state][action])
                    break
                else:
                    # Choose A' from S' using policy
                    next_action = self.choose_action(next_state)
                    
                    # SARSA update
                    self.Q[state][action] += self.alpha * (
                        reward + 
                        self.gamma * self.Q[next_state][next_action] - 
                        self.Q[state][action]
                    )
                    
                    # Move to next state
                    state, action = next_state, next_action
            
            self.episode_rewards.append(episode_reward)
            
            # Print progress
            if (ep + 1) % (self.episodes // 10) == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {ep+1}/{self.episodes} - Avg Reward: {avg_reward:.2f}")

        # Create deterministic policy from Q-values
        policy = {}
        for state, actions in self.Q.items():
            best_action = max(actions.items(), key=lambda x: x[1])[0]
            policy[state] = best_action

        return policy, self.Q
