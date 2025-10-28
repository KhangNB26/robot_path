# environment/gridworld.py
"""
GridWorld environment:
- grid: 0 empty, 1 obstacle, 2 goal cell
- start(s): four corners as possible starts; actual start chosen by main
- each goal cell stores number of items (default items_per_goal)
- ensures all goals are reachable from chosen start by regenerating map if necessary
"""

import numpy as np
from collections import deque
from utils import neighbors4, set_seed
import random

class GridWorld:
    def __init__(self, size, num_goal_cells, items_per_goal, obstacle_prob, seed=None):
        self.size = size
        self.num_goal_cells = num_goal_cells
        self.items_per_goal = items_per_goal
        self.obstacle_prob = obstacle_prob
        self.seed = set_seed(seed)
        self.reset()

    def corners(self):
        n = self.size
        return [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]

    def reset(self, chosen_start=None):
        # generate grid until all goals reachable
        tries = 0
        while True:
            tries += 1
            self.grid = np.zeros((self.size, self.size), dtype=int)
            # place random obstacles
            for r in range(self.size):
                for c in range(self.size):
                    if random.random() < self.obstacle_prob:
                        self.grid[r,c] = 1
            # choose start
            self.start = chosen_start or random.choice(self.corners())
            # pick goal cells (cannot be start)
            all_positions = [(r,c) for r in range(self.size) for c in range(self.size)]
            candidates = [p for p in all_positions if p not in [self.start]]
            goals = random.sample(candidates, self.num_goal_cells)
            self.goal_cells = {}
            for g in goals:
                r,c = g
                self.grid[r,c] = 2
                self.goal_cells[g] = self.items_per_goal
            # ensure corners are not obstacles
            for corner in self.corners():
                self.grid[corner] = 0

            # ensure reachability
            if self._all_goals_reachable():
                break
            if tries > 200:
                # adjust obstacle_prob to ensure feasible map
                self.obstacle_prob = max(0.0, self.obstacle_prob - 0.01)
        # internal state for simulation
        self.robot_pos = self.start
        self.carried = 0
        return self

    def _all_goals_reachable(self):
        # BFS from start, check all goal cells reachable
        visited = set()
        q = deque([self.start])
        visited.add(self.start)
        while q:
            pos = q.popleft()
            r,c = pos
            for nb in neighbors4(pos, (self.size, self.size)):
                if nb in visited:
                    continue
                if self.grid[nb] == 1:
                    continue
                visited.add(nb)
                q.append(nb)
        # check each goal cell
        for g in self.goal_cells.keys():
            if g not in visited:
                return False
        return True

    def is_obstacle(self, pos):
        return self.grid[pos] == 1

    def is_goal(self, pos):
        return pos in self.goal_cells and self.goal_cells[pos] > 0

    def pick_items(self, pos, amount):
        # pick up `amount` items from pos (a goal cell). Return actual picked.
        if pos not in self.goal_cells:
            return 0
        available = self.goal_cells[pos]
        picked = min(available, amount)
        self.goal_cells[pos] -= picked
        if self.goal_cells[pos] <= 0:
            # clear the grid marking (goal is empty)
            r,c = pos
            self.grid[r,c] = 0
        return picked

    def goals_remaining(self):
        total = sum(self.goal_cells.values())
        return total

    def copy(self):
        # shallow copy utility (for planners)
        new = GridWorld(self.size, self.num_goal_cells, self.items_per_goal, self.obstacle_prob, self.seed)
        new.grid = self.grid.copy()
        new.goal_cells = dict(self.goal_cells)
        new.start = self.start
        new.robot_pos = self.robot_pos
        new.carried = self.carried
        return new
