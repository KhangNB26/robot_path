# mdp/mdp_model.py
"""
A simple MDP model wrapper to be used by value/policy iteration.
State representation (for tabular RL): (r, c, carried, goals_state_key)
- goals_state_key: tuple of remaining items at each goal index (ordered)
Note: This MDP is for small-scale tabular methods; for more complex RL you'd need function approximation.
"""

import itertools
import numpy as np
from collections import OrderedDict
from utils import neighbors4, manhattan

class SimpleMDPModel:
    def __init__(self, gridworld, carry_capacity):
        """
        Build a model of the environment for tabular RL:
        - states: all combinations of robot pos and distribution of remaining items per goal
        - actions: up/down/left/right/stay/pick/drop
        Note: For tractability we will compress goal distribution as tuple of counts for goal cells.
        """
        self.gw = gridworld
        self.size = gridworld.size
        self.goal_positions = list(gridworld.goal_cells.keys())
        self.goal_initial = [gridworld.items_per_goal]*len(self.goal_positions)
        self.capacity = carry_capacity
        # build mapping from state tuple to index lazily in agents
        # we'll provide transition function instead of enumerating everything (could be huge)
        self.actions = [(1,0),(-1,0),(0,1),(0,-1),(0,0)]  # last is stay, pick/drop handled in step
        # reward scheme
        self.step_cost = -1.0
        self.pick_reward = 10.0
        self.return_reward = 20.0
        self.start = gridworld.start

    def is_terminal(self, goals_state):
        # terminal if all goals are satisfied and robot is at start
        return sum(goals_state) == 0

    def step(self, state, action):
        """
        state: (pos, carried, goals_state_tuple)
        action: delta (dr,dc) or 'pick'/'drop' handled externally if desired
        returns: next_state, reward
        """
        (r,c), carried, goals = state
        # movement
        dr, dc = action
        nr, nc = r + dr, c + dc
        # check bounds and obstacles
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            nr, nc = r, c  # invalid move - stay
        if self.gw.grid[nr, nc] == 1:
            nr, nc = r, c
        new_goals = list(goals)
        reward = self.step_cost
        # if standing on a goal cell and there are items, pick automatically up to capacity
        pos = (nr, nc)
        if pos in self.goal_positions:
            idx = self.goal_positions.index(pos)
            if new_goals[idx] > 0 and carried < self.capacity:
                pick = min(new_goals[idx], self.capacity - carried)
                new_goals[idx] -= pick
                carried += pick
                reward += pick * self.pick_reward
        # if at start and carrying > 0, return items (drop)
        if pos == self.start and carried > 0:
            reward += carried * self.return_reward
            carried = 0
        return ((nr, nc), carried, tuple(new_goals)), reward
    def get_all_states(self):
        """
        Generate all possible states for tabular RL.
        State: (position, carried_items, goals_state)
        """
        # Get all valid positions (non-obstacle cells)
        positions = []
        for r in range(self.size):
            for c in range(self.size):
                if self.gw.grid[r,c] != 1:  # not obstacle
                    positions.append((r,c))
                    
        # All possible carried amounts
        carried_amounts = range(self.capacity + 1)
        
        # All possible goal states
        # For each goal, items can range from 0 to initial amount
        goal_ranges = [range(g + 1) for g in self.goal_initial]
        goal_states = list(itertools.product(*goal_ranges))
        
        # Generate all combinations
        states = []
        for pos in positions:
            for carried in carried_amounts:
                for goals in goal_states:
                    states.append((pos, carried, goals))
                    
        return states
