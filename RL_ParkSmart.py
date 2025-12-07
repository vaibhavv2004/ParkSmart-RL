import pygame
import random
import numpy as np
import math
from stable_baselines3 import PPO
from gymnasium import Env, spaces
from collections import deque

# ---------------- 1. Custom Parking Environment (Unchanged Core Logic) ----------------
class ParkingEnv(Env):
    # Colors for observation normalization
    MAX_COST = 5.0
    MAX_DISTANCE = 50.0 # Based on average parking lot size

    def __init__(self, num_slots=12, render_mode=None):
        super().__init__()
        self.num_slots = num_slots
        self.slots = []
        self.vehicle_pos = np.array([50.0, 450.0])
        self.selected_slot_index = None
        self.current_episode_reward = 0.0

        # Define Observation and Action Space
        # Observation: normalized cost, distance, occupied (1.0 or 0.0)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(num_slots, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(num_slots)
        self.render_mode = render_mode
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.slots = []
        self.vehicle_pos = np.array([50.0, 450.0])
        self.current_episode_reward = 0.0
        
        for _ in range(self.num_slots):
            # Slot positions are for the visualization, not strictly part of the RL state
            x = random.randint(100, 800)
            y = random.randint(50, 300)
            occupied = random.choice([False, False, True]) # 1/3 chance of occupied
            cost = random.choice([1, 3, 5])
            
            dx = x + 40 - self.vehicle_pos[0] # To center of slot
            dy = y + 25 - self.vehicle_pos[1]
            distance = int(np.hypot(dx, dy) / 10)
            
            # Simplified reward: penalized heavily for occupied, penalized by cost/distance
            reward = -200 if occupied else (10 - cost) - distance
            
            self.slots.append({
                "pos": (x, y),
                "occupied": occupied,
                "cost": cost,
                "distance": distance,
                "reward": reward
            })
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        # Creates the observation array for the RL agent
        obs = np.zeros((self.num_slots, 3), dtype=np.float32)
        for i, slot in enumerate(self.slots):
            obs[i, 0] = slot["cost"] / self.MAX_COST      # Normalized Cost
            obs[i, 1] = slot["distance"] / self.MAX_DISTANCE # Normalized Distance
            obs[i, 2] = float(slot["occupied"])             # Occupied Status
        return obs

    def _get_info(self):
        return {"selected_slot_index": self.selected_slot_index}

    def step(self, action):
        self.selected_slot_index = action
        slot = self.slots[action]
        reward = slot["reward"]
        
        # In this simple model, the episode ends after one decision
        terminated = True
        truncated = False
        
        self.current_episode_reward += reward
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

# ---------------- 2. Pygame Visualization Layer ----------------
class ParkingVisualization:
    # Colors
    FREE = (144, 238, 144)
    OCCUPIED = (255, 182, 193)
    SELECTED = (173, 216, 230)
    VEHICLE = (255, 215, 0)
    ROAD = (100, 100, 100)
    TEXT_COLOR = (0, 0, 0)
    
    # Dimensions
    SCREEN_WIDTH = 900
    SCREEN_HEIGHT = 500
    SLOT_W = 80
    SLOT_H = 50

    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("ParkSmart RL Visualization")
        self.font = pygame.font.SysFont(None, 24)
        self.clock = pygame.time.Clock()
        
        self.vehicle_path = deque(maxlen=50) # Stores last 50 vehicle positions for path tracing
        self.episode_count = 0
        self.total_reward_history = []
        self.current_reward = 0.0

    def get_color_from_reward(self, reward):
        """Maps a reward value to a color gradient (Heatmap logic)"""
        # Define Min/Max Reward for the gradient (e.g., -50 to +10)
        # Based on reward formula: max(10-1-0) = 9, min(-200)
        MIN_R = -200
        MAX_R = 10
        
        normalized = (reward - MIN_R) / (MAX_R - MIN_R)
        
        # Clamp to ensure it stays between 0 and 1
        normalized = max(0, min(1, normalized))
        
        # Color transition from Red (bad reward) to Green (good reward)
        R = int(255 * (1 - normalized))
        G = int(255 * normalized)
        B = 0
        return (R, G, B)

    def draw_parking_lot(self, selected_slot_index, q_values=None):
        self.screen.fill((200, 200, 200)) # Grey background for the lot area
        
        # Draw road
        pygame.draw.rect(self.screen, self.ROAD, (0, 400, self.SCREEN_WIDTH, 100))
        
        # Draw path trace
        for i in range(1, len(self.vehicle_path)):
            pos1 = self.vehicle_path[i-1]
            pos2 = self.vehicle_path[i]
            # Fade the color based on how old the trace point is
            alpha = int(255 * (i / len(self.vehicle_path)))
            color = (*self.VEHICLE[:3], alpha)
            pygame.draw.line(self.screen, color, pos1, pos2, 3)

        # Draw slots
        for i, slot in enumerate(self.env.slots):
            x, y = slot["pos"]
            
            # --- 1. HEATMAP (Value Visualization) ---
            if q_values is not None:
                # Assuming the PPO agent's output can be mapped to a value function V(s) or Q(s,a)
                # For simplicity, we use the raw slot reward for color, but a better implementation 
                # would use the agent's learned Q-value for that slot.
                color = self.get_color_from_reward(slot["reward"])
                pygame.draw.rect(self.screen, color, (x, y, self.SLOT_W, self.SLOT_H))
                
                # Draw a white outline on the heatmap to define the slot
                pygame.draw.rect(self.screen, (255, 255, 255), (x, y, self.SLOT_W, self.SLOT_H), 1)
            else:
                 # Default color for occupied/free state if Q-values aren't used
                color = self.OCCUPIED if slot["occupied"] else self.FREE
                pygame.draw.rect(self.screen, color, (x, y, self.SLOT_W, self.SLOT_H))
                pygame.draw.rect(self.screen, self.TEXT_COLOR, (x, y, self.SLOT_W, self.SLOT_H), 2)


            # --- 2. SELECTED SLOT HIGHLIGHT ---
            if selected_slot_index == i:
                pygame.draw.rect(self.screen, self.SELECTED, (x, y, self.SLOT_W, self.SLOT_H), 4)

            # --- 3. Slot Data Display ---
            self.screen.blit(self.font.render(f"${slot['cost']}", True, self.TEXT_COLOR), (x + 5, y + 5))
            self.screen.blit(self.font.render(f"D:{slot['distance']}", True, self.TEXT_COLOR), (x + 5, y + 20))
            self.screen.blit(self.font.render(f"R:{slot['reward']}", True, self.TEXT_COLOR), (x + 5, y + 35))
            self.screen.blit(self.font.render(f"Slot {i}", True, self.TEXT_COLOR), (x + 45, y + 5))
            
        # Draw vehicle
        self.vehicle_path.append(tuple(self.env.vehicle_pos))
        pygame.draw.circle(self.screen, self.VEHICLE, (int(self.env.vehicle_pos[0]), int(self.env.vehicle_pos[1])), 20)

    def draw_info(self, selected_slot_index):
        # Controls/Status
        self.screen.blit(self.font.render("Controls: SPACE=PPO Select, A=Auto, R=Reset, ESC=Exit", True, self.TEXT_COLOR), (10, 10))
        
        # Agent Performance
        self.screen.blit(self.font.render(f"Episode: {self.episode_count}", True, self.TEXT_COLOR), (10, 30))
        self.screen.blit(self.font.render(f"Last Reward: {self.current_reward:.2f}", True, self.TEXT_COLOR), (10, 50))
        
        # Selected Slot Info
        if selected_slot_index is not None:
            slot = self.env.slots[selected_slot_index]
            text = f"Selected Slot {selected_slot_index}: Cost:${slot['cost']} Dist:{slot['distance']} Final Reward:{slot['reward']}"
            self.screen.blit(self.font.render(text, True, self.TEXT_COLOR), (10, 70))

    def move_vehicle(self, target_pos, speed=5):
        """Moves the vehicle towards the target position."""
        if target_pos is None:
            return True # Not moving
            
        target_pos_np = np.array(target_pos)
        direction = target_pos_np - self.env.vehicle_pos
        dist = np.linalg.norm(direction)
        
        if dist < speed:
            self.env.vehicle_pos = target_pos_np
            return True # Reached target
            
        self.env.vehicle_pos += speed * direction / dist
        return False # Still moving

# ---------------- 3. Main Logic and Training ----------------
# Setup
env = ParkingEnv(num_slots=12)
vis = ParkingVisualization(env)
agent = PPO("MlpPolicy", env, verbose=0)

# Training (Can be moved outside the main loop for faster startup)
print("Training PPO agent for 5000 timesteps...")
agent.learn(total_timesteps=15000, progress_bar=True)
print("Training complete.")

# Loop variables
obs, _ = env.reset()
selected_slot_index = None
auto_mode = False
moving = False
target_pos = None

def execute_ppo_action(current_obs):
    """Makes the agent select an action and sets up the movement."""
    action, _ = agent.predict(current_obs, deterministic=True)
    
    # Execute the action in the environment to get the reward
    new_obs, reward, terminated, truncated, info = env.step(action.item())
    
    # Set up visualization variables
    vis.current_reward = reward
    vis.episode_count += 1
    
    slot = env.slots[action.item()]
    target = [slot["pos"][0] + vis.SLOT_W/2, slot["pos"][1] + vis.SLOT_H/2]
    
    return action.item(), target, new_obs, terminated or truncated


# --- Main Pygame Loop ---
running = True
while running:
    # 1. Input Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                obs, _ = env.reset()
                selected_slot_index = None
                moving = False
                target_pos = None
                vis.vehicle_path.clear()
            elif event.key == pygame.K_SPACE and not moving:
                selected_slot_index, target_pos, obs, done = execute_ppo_action(obs)
                moving = True
            elif event.key == pygame.K_a:
                auto_mode = not auto_mode
                print(f"Auto Mode: {'ON' if auto_mode else 'OFF'}")

    # 2. Auto-Mode Logic
    if auto_mode and not moving:
        selected_slot_index, target_pos, obs, done = execute_ppo_action(obs)
        moving = True

    # 3. Vehicle Movement
    if moving:
        done_moving = vis.move_vehicle(target_pos)
        if done_moving:
            moving = False
            # If done, immediately reset the environment for the next episode
            if auto_mode:
                obs, _ = env.reset()
                selected_slot_index = None
                vis.vehicle_path.clear()

    # 4. Drawing
    vis.draw_parking_lot(selected_slot_index) # Passes None for Q-values, can be added later
    vis.draw_info(selected_slot_index)
    
    pygame.display.flip()
    vis.clock.tick(30)

pygame.quit()




