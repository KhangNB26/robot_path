# main.py
"""
Main runner:
- Builds gridworld (from config.yaml)
- Demonstrates planners for one trip: plan route from start to one goal and back
- Optionally visualizes with pygame_viz
- Runs simple RL agent (Value Iteration) demo to compute policy (for small grids)
"""
import yaml
import os
from env.gridworld import GridWorld
from visualization.pygame_viz import animate_path
from mdp.mdp_model import SimpleMDPModel
from rl_agents.value_iteration import ValueIterationAgent
from rl_agents.policy_iteration import PolicyIterationAgent
from rl_agents.q_learning import QLearningAgent
from rl_agents.sarsa import SarsaAgent
from rl_agents.td0 import TD0Agent
from rl_agents.td_lambda import TDLambdaAgent
from planners.bfs import bfs_grid
from utils import set_seed, timer, timing 
import time
import csv 

def load_config(path=None):
    if path is None:
        # Use config.yaml in the same directory structure
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(current_dir, "config", "config.yaml")
    with open(path,"r",encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def create_agent(agent_type, mdp, cfg):
    """Create RL agent based on type and config"""
    gamma = cfg.get("gamma")
    theta = cfg.get("theta")
    max_iters = cfg.get("max_iters")
    # max_states = cfg.get("max_states")
    max_steps = cfg.get("max_steps")
    alpha = cfg.get("alpha")
    epsilon = cfg.get("epsilon")
    episodes = cfg.get("episodes")
    # epsilon_decay = cfg.get("epsilon_decay")
    lam = cfg.get("lam")

    if agent_type == "value_iteration":
        return ValueIterationAgent(mdp, gamma, theta, max_iters)
    elif agent_type == "policy_iteration":
        return PolicyIterationAgent(mdp, gamma, max_iters)
    elif agent_type == "q_learning":
        return QLearningAgent(mdp, alpha, gamma, epsilon, episodes, max_steps)
    elif agent_type == "sarsa":
        return SarsaAgent(mdp, alpha, gamma, epsilon, episodes, max_steps)
    elif agent_type == "td0":
        return TD0Agent(mdp, alpha, gamma, episodes, max_steps)
    elif agent_type == "td_lambda":
        return TDLambdaAgent(mdp, alpha, gamma, lam, episodes, max_steps)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
def display_menu(agents):
    """Display menu for agent selection"""
    print("\nAvailable Agents:")
    print("================")
    for idx, agent in enumerate(agents, 1):
        print(f"{idx}. {agent.replace('_', ' ').title()}")
    print("0. Exit")


def run_rl_demo(cfg):
    # Set random seed
    set_seed(cfg.get("random_seed", None))
    
    # Create gridworld
    gw = GridWorld(size=cfg.get("grid_size"),
                  num_goal_cells=cfg.get("num_goal_cells"), # Reduced for RL
                  items_per_goal=cfg.get("items_per_goal"),
                  obstacle_prob=cfg.get("obstacle_prob"),
                  seed=cfg.get("random_seed", None))
    
    print("Start:", gw.start)
    print("Goals:", list(gw.goal_cells.items())[:6], " total items:", gw.goals_remaining())

    # Create MDP model and run Value Iteration
    mdp = SimpleMDPModel(gw, carry_capacity=cfg.get("carry_capacity"))

    # Agent selection menu
    available_agents = cfg.get("agents", {}).get("available", ["value_iteration"])
    default_agent = cfg.get("agents", {}).get("default", "value_iteration")
    
    while True:
        display_menu(available_agents)
        choice = input(f"\nSelect agent (1-{len(available_agents)}, default={default_agent}): ").strip()
        
        if choice == "0":
            print("Exiting...")
            return
        elif choice == "":
            agent_name = default_agent
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(available_agents):
            agent_name = available_agents[int(choice)-1]
            break
        else:
            print("Invalid choice. Please try again.")

    print(f"\nSelected agent: {agent_name.replace('_', ' ').title()}")
    
    # Create selected agent
    agent = create_agent(agent_name, mdp, cfg)
    
    # Initial state
    goals_state = tuple([gw.items_per_goal]*len(mdp.goal_positions))
    start_state = (gw.start, 0, goals_state)
    
    # Run agent
    print(f"\nRunning {agent_name.replace('_', ' ').title()}...")
    with timing(f"{agent_name} execution"):
        start_time = time.perf_counter()
        pi, V = agent.run(start_state)
        elapsed_time = time.perf_counter() - start_time
        # Check policy type and print appropriate message
    if isinstance(pi, dict):
        print(f"Training complete. Policy size: {len(pi)}")
    else:
        print("Training complete. Using function-based policy")

    # Generate path using policy
    if cfg.get("visualize", True):
        current_state = start_state
        path = [current_state[0]]
        rewards = []  # Track rewards
        carried_items = [current_state[1]]  # Track items carried at each step
        goal_states = [current_state[2]]  # Track goal states at each step
        steps = 0
        max_steps = cfg.get("max_steps")
        
        # Create a copy of gridworld for simulation
        sim_gw = gw.copy()
        
        while steps < max_steps:
            # Check if we've collected all items and returned to start
            if mdp.is_terminal(current_state[2]) and current_state[0] == gw.start:
                print("All items collected and returned to start.")
                break
            
            # If all items collected but not at start, force navigation back to start
            if mdp.is_terminal(current_state[2]) and current_state[0] != gw.start:
                print(f"All items collected at {current_state[0]}. Navigating back to start...")
                # Use BFS to find path back to start
                return_path = bfs_grid(sim_gw.grid, current_state[0], gw.start)
                if return_path and len(return_path) > 1:
                    # Follow the path back to start
                    for i in range(1, len(return_path)):
                        from_pos = return_path[i-1]
                        to_pos = return_path[i]
                        action = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
                        
                        # Update state
                        next_state, reward = mdp.step(current_state, action)
                        rewards.append(reward)
                        
                        pos, carried, goals = next_state
                        for idx, goal_pos in enumerate(mdp.goal_positions):
                            if goal_pos in sim_gw.goal_cells:
                                sim_gw.goal_cells[goal_pos] = goals[idx]
                        
                        current_state = next_state
                        path.append(current_state[0])
                        carried_items.append(current_state[1])
                        goal_states.append(current_state[2])
                        steps += 1
                    
                    print("Returned to start successfully!")
                    break
                else:
                    print("Could not find path back to start.")
                    break

            # Get action from policy (handle both dictionary and function policies)
            if isinstance(pi, dict):
                action = pi.get(current_state)
                if action is None:
                    print("No policy action found. Stopping.")
                    break
            else:
                action = pi(current_state)
                
            # Get next state and reward
            next_state, reward = mdp.step(current_state, action)
            rewards.append(reward)
            
            # Update gridworld state to match MDP state
            pos, carried, goals = next_state
            for idx, goal_pos in enumerate(mdp.goal_positions):
                if goal_pos in sim_gw.goal_cells:
                    sim_gw.goal_cells[goal_pos] = goals[idx]
                    # Keep the grid cell marked as goal (2) even when empty
                    # The visualization will handle showing it differently
            
            current_state = next_state
            path.append(current_state[0])
            carried_items.append(current_state[1])  # Track carried items
            goal_states.append(current_state[2])  # Track goal states
            steps += 1
        
        print(f"Path length: {len(path)}")
        animate_path(sim_gw, path, 
                    fps=cfg.get("render_fps"),
                    step_delay=cfg.get("step_delay"),
                    rewards=rewards,
                    carried_items=carried_items,
                    goal_states=goal_states,
                    mdp=mdp)  # Pass goal states and mdp to visualization
    
    log_path = "results/rl_metrics.csv"
    os.makedirs("results", exist_ok=True)

    file_exists = os.path.isfile(log_path)

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["agent", "total_reward", "total_steps", "time_execution"])
        writer.writerow([agent_name, sum(rewards), steps, elapsed_time])

    print(f"\nLogged results to {log_path}")
    print("Demo finished.")

if __name__ == "__main__":
    cfg = load_config()
    run_rl_demo(cfg)
