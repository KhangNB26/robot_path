"""
Simple Pygame visualization for grid and a given path (list of positions).
Call draw_run(gridworld, paths_list, start) where paths_list is list of waypoints to animate.
"""
import pygame
import sys
import time

CELL = 40
MARGIN = 2
COLORS = {
    'bg': (30,30,30),
    'empty': (220,220,220),
    'obstacle': (40,40,40),
    'goal': (191, 143, 0),
    'goal_empty': (130, 100, 50),  # Darker orange for empty goal cells
    'start': (30,144,255),
    'robot': (255,50,50),
    'path': (100,255,100),
    'text': (240,240,240)
}

def draw_grid(surface, gw, cellsize):
    rows, cols = gw.grid.shape
    font = pygame.font.SysFont(None, 24)  # Font for item counts

    for r in range(rows):
        for c in range(cols):
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            rect = pygame.Rect(x, y, cellsize, cellsize)

            if gw.grid[r,c] == 1:
                color = COLORS['obstacle']
            elif (r,c) == gw.start:
                color = COLORS['start']
            elif (r,c) in gw.goal_cells:
                if gw.goal_cells[(r,c)] > 0:
                    color = COLORS['goal']  # Orange for goals with items
                else:
                    color = COLORS['goal_empty']  # Darker orange for empty goals
            else:
                color = COLORS['empty']
            
            pygame.draw.rect(surface, color, rect)

            # Draw items count for goal cells with remaining items
            if (r,c) in gw.goal_cells and gw.goal_cells[(r,c)] > 0:
                items = str(gw.goal_cells[(r,c)])
                text = font.render(items, True, COLORS['text'])
                text_rect = text.get_rect(center=rect.center)
                surface.blit(text, text_rect)

def animate_path(gw, path, fps=2, title="Robot", step_delay=5.0, rewards=None, carried_items=None, goal_states=None, mdp=None):
    """
    rewards: optional list of reward values aligned with transitions (len = len(path)-1)
    carried_items: optional list of items carried at each step (len = len(path))
    goal_states: optional list of goal state tuples at each step (len = len(path))
    mdp: MDP model to map goal states to positions
    """
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont(None, 20)
    robot_font = pygame.font.SysFont(None, 28, bold=True)  # Font for items on robot
    rows, cols = gw.grid.shape
    cellsize = CELL
    width = cols * (cellsize + MARGIN) + MARGIN
    height = rows * (cellsize + MARGIN) + MARGIN + 60
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    running = True
    step = 0
    total_reward = 0
    initial_items = sum(gw.goal_cells.values())  # Track initial total items
    remaining_items = initial_items  # Track remaining items
    items_delivered = 0  # Track items actually delivered to start
    prev_carried = 0  # Track previous carried to detect delivery

    while running:
        clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(COLORS['bg'])
        
        # Update goal cells for current step if goal_states provided
        if goal_states and mdp and step < len(goal_states):
            current_goals = goal_states[step]
            for idx, goal_pos in enumerate(mdp.goal_positions):
                if goal_pos in gw.goal_cells:
                    gw.goal_cells[goal_pos] = current_goals[idx]
        
        # Draw path up to current step (BEFORE grid so it doesn't cover numbers)
        for i, p in enumerate(path[:step+1]):
            r, c = p
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            center = (x + cellsize//2, y + cellsize//2)
            # Draw smaller path dots that won't cover text
            pygame.draw.circle(screen, COLORS['path'], center, cellsize//8)
        
        draw_grid(screen, gw, cellsize)

        # Draw robot at current position
        if step < len(path):
            r, c = path[step]
            x = c * (cellsize + MARGIN) + MARGIN
            y = r * (cellsize + MARGIN) + MARGIN
            rect = pygame.Rect(x+4, y+4, cellsize-8, cellsize-8)
            pygame.draw.rect(screen, COLORS['robot'], rect)
            
            # Draw number of items carried on the robot
            if carried_items and step < len(carried_items):
                items_carrying = carried_items[step]
                if items_carrying > 0:
                    # Draw a small circle or badge with the number
                    badge_text = robot_font.render(str(items_carrying), True, (255, 255, 255))
                    badge_rect = badge_text.get_rect(center=(x + cellsize//2, y + cellsize//2))
                    # Draw a dark background circle for better visibility
                    pygame.draw.circle(screen, (0, 0, 0), 
                                     (x + cellsize//2, y + cellsize//2), 
                                     cellsize//4)
                    screen.blit(badge_text, badge_rect)

        # Update statistics
        current_reward = 0
        if rewards and step-1 >= 0 and step-1 < len(rewards):
            current_reward = rewards[step-1]
            total_reward += current_reward
            
        # Calculate remaining items on map
        remaining_items = sum(gw.goal_cells.values())
        
        # Get current carried items
        current_carried = 0
        if carried_items and step < len(carried_items):
            current_carried = carried_items[step]
            
            # Detect delivery: if robot was carrying items and now carries less, and is at start
            if step > 0 and path[step] == gw.start:
                prev_carried = carried_items[step-1]
                if prev_carried > current_carried:
                    items_delivered += (prev_carried - current_carried)

        # Update display information
        info_lines = [
            f"Step: {step}/{len(path)-1}",
            f"Current Reward: {current_reward:+.1f}",
            f"Total Reward: {total_reward:+.1f}",
            f"Carrying: {current_carried} items",
            f"Items Delivered: {items_delivered}/{initial_items}",
            f"Remaining on Map: {remaining_items}"
        ]

        # Draw info text
        for i, line in enumerate(info_lines):
            txt = font.render(line, True, COLORS['text'])
            screen.blit(txt, (10, height-80 + i*18))

        pygame.display.flip()
        step += 1

        # Check if we've reached the end of path
        if step >= len(path):
            # Show final state and wait for user to close
            print("\n" + "="*50)
            print("Animation complete!")
            print("Final Statistics:")
            print(f"  - Total Steps: {len(path)-1}")
            print(f"  - Total Reward: {total_reward:.1f}")
            print(f"  - Items Delivered: {items_delivered}/{initial_items}")
            print("="*50)
            print("\nWindow will stay open. Close it manually when done.")
            
            # Keep window open until user closes it
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            waiting = False
                pygame.display.flip()
                clock.tick(10)  # Lower CPU usage while waiting
            
            running = False

    pygame.quit()
