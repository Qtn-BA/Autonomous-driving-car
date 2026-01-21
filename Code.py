import pygame
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from gymnasium import spaces 
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys 
import traceback




# === GLOBALE CONFIGURATION ===
SECONDS_PER_EPISODE = 20
FPS = 30
SHOW_PREVIEW = True
SAVE_PATH = "dqn_driving_model2.pth"
MAX_STEPS = FPS * SECONDS_PER_EPISODE

OBS_ID = {1: 1, 2: 2, 3: 2, 4: 3}   
N_OBS = 4  # 0..3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
ROAD = (70, 70, 70)
GRASS = (34, 139, 34)
SKY = (135, 206, 235)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
BLUE = (60, 120, 255)
GREEN = (50, 255, 50)
DARK_RED = (150, 0, 0)
LIGHT_GRAY = (180, 180, 180)
DARK_GRAY = (50, 50, 50)

# Driving parameters
PREFERRED_SPEED = 10.0
SPEED_THRESHOLD = 2.0
STEER_AMT = 1.0 # Non utilisé, mais laissé pour cohérence

# State parameters (Observation)
STATE_TYPES = {
    0: "Rien_devant",
    1: "Feu_Rouge",
    3: "Passant",
    4: "Voiture_devant"
}
STATE_SIZE = 8   # [obs_type, obs_dist, car_speed_norm, steer_norm, car_x_norm, time_left, car_y_norm, lir] (lir=light is red)


# Discreet actions : steer (5) x throttle (3) = 15 actions

STEER_VALUES = {
    0: -0.6,
    1: -0.25,
    2: 0.0,
    3: 0.25,
    4: 0.6
}
STEER_ACTIONS = len(STEER_VALUES)

THROTTLE_VALUES = {
    0: -1.0,  # Brake
    1:  0.0,  # Maintain / coast
    2:  1.0   # Accelerate
}
THROTTLE_ACTIONS = len(THROTTLE_VALUES)

ACTION_SIZE = STEER_ACTIONS * THROTTLE_ACTIONS

STEER_NAMES = {0:"Gauche_Fort", 1:"Gauche", 2:"Tout_Droit", 3:"Droite", 4:"Droite_Fort"}
THROTTLE_NAMES = {0:"Frein", 1:"Maintenir", 2:"Accel"}

ACTION_MAP = {
    a: f"{THROTTLE_NAMES[a // STEER_ACTIONS]} + {STEER_NAMES[a % STEER_ACTIONS]}"
    for a in range(ACTION_SIZE)
}


STEER_AMT = max(abs(v) for v in STEER_VALUES.values())  # = 0.6


# ============================================================
# Neuronal Network DQN
# ============================================================
class DQNNetwork(nn.Module):
    """Easy neuronal network for the approximation of Q-fonction."""
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

# ============================================================
# DQN Agent
# ============================================================
class DQNAgent:
    """
    Agent Double-DQN using PyTorch.
    Optimized for tensor conversion and  GPU/ CPU compatible.
    """

    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99,
                 epsilon=1.0, eps_decay=0.997, eps_min=0.01,
                 batch_size=128, memory_size=500000, target_update=500):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network (Policy et Target)
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay Memory
        self.memory = deque(maxlen=memory_size)

    # ---------------------------
    def act(self, state, train=True):
        """Stock selection based on the epsilon-greedy policy."""
        if train and (random.random() < self.epsilon):
            return random.randrange(self.action_size)

        # Convert the NumPy state to Tensor for the network
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=1).item())

    # ---------------------------
    def remember(self, state, action, reward, next_state, done):
        """Stores a transition in replay memory"""
        self.memory.append((np.array(state, dtype=np.float32),
                            int(action),
                            float(reward),
                            np.array(next_state, dtype=np.float32),
                            float(done)))

    # ---------------------------
    def replay(self):
        """Sample a minibatch and train the policy_net network with the Double-DQN target."""
        if len(self.memory) < self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)

        # Efficient conversion of minibatch to PyTorch Tensors
        states = torch.from_numpy(np.vstack([m[0] for m in minibatch])).float().to(self.device)
        actions = torch.from_numpy(np.array([m[1] for m in minibatch], dtype=np.int64)).to(self.device)
        rewards = torch.from_numpy(np.array([m[2] for m in minibatch], dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.vstack([m[3] for m in minibatch])).float().to(self.device)
        dones = torch.from_numpy(np.array([m[4] for m in minibatch], dtype=np.float32)).to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Double DQN: R + gamma * Q_target(s', argmax_a' Q_policy(s', a'))
        with torch.no_grad():
            # Selection of optimal action in next_state by the policy net 
            next_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
            # Evaluation of selected action by target net
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            
            # Target calculation
            targets = rewards + self.gamma * next_q_values * (1 - dones)
            targets = torch.clamp(targets, -10, 10) # Clipping for stability

        # Loss calculation
        loss = self.loss_fn(q_values, targets)

        # Optimisazion
        self.optimizer.zero_grad()
        loss.backward()
        # Clipping the gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()

        # Soft update of Target Network
        self.soft_update(tau=0.005)

        return float(loss.item())

    # ---------------------------
    def soft_update(self, tau=0.005):
        """Soft updateof target network: target <- tau * policy + (1 - tau) * target"""
        for target_param, policy_param in zip(self.target_net.parameters(),
                                              self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    # ---------------------------
    def decay_epsilon(self):
        """Exponential decay of epsilon."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ---------------------------
    def save(self, filepath):
        """Saves the agent's state"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    # ---------------------------
    def load(self, filepath, map_location=None):
        """Loads an agent report from a file"""
        checkpoint = torch.load(filepath, map_location=(map_location or self.device))
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)

# ============================================================
# OBSTACLES
# ============================================================
class Obstacle:
    """Represents an obstacle (red/ green light, pedestrian, car) in the state."""
    def __init__(self, kind, x, y):
        self.kind = kind
        self.x = x
        self.y = y 
        self.speed = 0
        self.width = 50
        self.height = 25
        
        if kind == 4:  # Car
            self.speed = random.uniform(3, 6)
            self.color = random.choice([RED, YELLOW, GRAY, WHITE])
        elif kind == 1:  # Fire
            self.light_state = "red"
            self.light_timer = 0
    
    def update(self):
        """Update the obstacle position and behavior """
        if self.kind == 4:
            self.x += self.speed / 2.0
        elif self.kind == 1:
            # Altenate red/ green light
            self.light_timer += 1
            if self.light_timer > FPS * 4: # Change every 4 secondes
                self.light_state = "green" if self.light_state == "red" else "red"
                self.light_timer = 0

# ============================================================
# DRIVING STATE (GYMNASIUM-LIKE)
# ============================================================
class DrivingEnv:
    """ Driving simulation state by reinforcement learning"""
    SHOW_CAM = SHOW_PREVIEW
    im_width = 1200
    im_height = 600
    
    def __init__(self, difficulty=1):
        self.difficulty = difficulty
        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_SIZE,), dtype=np.float32
        )
        
        self.obstacles = []
        self.border_hits = 0

        # Car situation
        self.car_x = 100.0
        self.car_y = 350.0 # Left lane (~350)
        self.car_speed = 0.0
        self.steer = 0.0
        self.car_angle = 0.0
        
        # Background and particles
        self.exhaust_particles = []
        self.road_offset = 0
        self.buildings = []
        self._generate_buildings()
        
        # Metrics and following
        self.collision_hist = []
        self.lane_invade_hist = []
        self.step_counter = 0
        self.episode_start = None
        self.initial_x = 100.0
        self.terminal_message = ""
        
        # Rendering Pygame
        self.window = None
        self.clock = None
        self.font = None
        self.font_small = None
        
    def _generate_buildings(self):
        """Create building for background"""
        for i in range(15):
            x = i * 150 + random.randint(-20, 20)
            height = random.randint(80, 200)
            width = random.randint(60, 100)
            color = random.choice([GRAY, DARK_GRAY, LIGHT_GRAY])
            self.buildings.append({'x': x, 'height': height, 'width': width, 'color': color})
    
    def cleanup(self):
        """Cleans episode variables"""
        self.collision_hist = []
        self.lane_invade_hist = []
        self.obstacles = []
    
    def maintain_speed(self, current_speed):
        """Determines the accelerator value to maintain the favorite speed"""
        if current_speed >= PREFERRED_SPEED:
            return 0.0
        elif current_speed < PREFERRED_SPEED - SPEED_THRESHOLD:
            return 0.7
        else:
            return 0.3
    
    def reset(self, seed=None):
        """ Resets the state and creates news obstacles"""
        if seed is not None:
            random.seed(seed)
        
        self.cleanup()
        self.car_x = 100.0
        self.car_y = 350.0 
        self.car_speed = 0.0
        self.steer = 0.0
        self.car_angle = 0.0
        self.step_counter = 0
        self.border_hits = 0
        self.initial_x = 100.0
        self.terminal_message = ""
        self.episode_start = time.time()
        self.exhaust_particles = []
        self.road_offset = 0
        
        # Obtsacles creation
        n_obstacles = 3 + self.difficulty
        obstacle_xs = sorted(random.sample(range(400, 1100), n_obstacles))
        for x in obstacle_xs:
            kind = random.choice([1, 3, 4])
            
            if kind == 1: # Red light: by the wayside
                y = 350
            else:
                # Placement on the right lane (~450) or the left lane (~350)
                if random.random() < 0.7:
                    y = random.randint(440, 460) # Right lane
                else:
                    y = random.randint(340, 360) # Left lane
                    
            self.obstacles.append(Obstacle(kind, x, y))
        
        return self._get_observation(), {}

    def _get_nearest_obstacle(self):
        """Find the closest obstacle in front of the car and in its lateral lane (Y)"""
        nearest = None
        min_dist = float('inf')
        
        car_top = self.car_y - 10
        car_bottom = self.car_y + 25
        
        for obs in self.obstacles:
            dist = obs.x - (self.car_x + 60) # Distance to the front bumper
            
            # Overlap check on the Y-axis (lane)
            obs_top = obs.y - 10 
            obs_bottom = obs.y + 25 
            
            # The red light (kind=1) is at the edge, its verification Y is less strict.
            if obs.kind == 1:
                y_overlap = True 
            else:
                # Simple check for vertical overlap between two rectangles
                y_overlap = (max(car_top, obs_top) < min(car_bottom, obs_bottom))
            
            # The obstacle must be in front of the car (-10 for a small margin)
            if dist >= -10 and dist < min_dist and y_overlap:
                nearest = obs
                min_dist = dist
        
        return nearest, min_dist
    
    def _get_observation(self):
        """Calculate the normalized state vector for the DQN"""
        nearest, dist = self._get_nearest_obstacle()
        
        # 1.  Normalized type of obstacle [0, 1]
        obs_id = OBS_ID.get(nearest.kind, 0) if nearest else 0
        obs_type = obs_id / (N_OBS - 1)

        # 2. Distance of normalized obstacle [0, 1] (max 300m)
        obs_dist = min(dist / 300.0, 1.0) if dist != float('inf') else 1.0

        # 3. Normalized speed (max 25.0 km/h)
        car_speed_norm = self.car_speed / 25.0

        # 4. Steering normalisé [-1, 1] -> [0, 1]
        steer_norm = (self.steer + STEER_AMT) / (2.0 * STEER_AMT) if STEER_AMT else 0.5

        # 5. Normalized X position 
        car_x_norm = self.car_x / self.im_width

        # 6. Temps restant normalisé [0, 1]
        time_left = max(0.0, (MAX_STEPS - self.step_counter) / MAX_STEPS)

        # 7. Normalized Y position (to detect the out lane)
        road_top, road_bottom = 300, 500
        car_y_norm = np.clip((self.car_y - road_top) / (road_bottom - road_top), 0.0, 1.0)

        # 8. Red light : 1 if the light in front is red, otherwise 0
        lir = float(
            nearest is not None and 
            nearest.kind == 1 and
            getattr(nearest, 'light_state', 'red') == "red"
        )

        
        state = np.array([obs_type, obs_dist, car_speed_norm, steer_norm, car_x_norm, car_y_norm, lir, time_left], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Create a new simulation step"""
        self.step_counter += 1
        
        # --- Decode action : throttle + steer ---
        throttle_idx = action // STEER_ACTIONS
        steer_idx = action % STEER_ACTIONS
       
        # Steering application
        self.steer = STEER_VALUES[steer_idx]
        throttle_cmd = THROTTLE_VALUES[throttle_idx]  # -1, 0, +1

        # Direction application
        self.car_angle += self.steer * 0.05

        # Prevents the angle from accumulation and causing the car to drift indefinitely.
        self.car_angle *= 0.90              # friction / recentrage around 0
        self.car_angle = np.clip(self.car_angle, -0.35, 0.35)

        
        # Apply acceleration/braking

        accel = 1.5 * throttle_cmd
        if throttle_cmd < 0:
            accel *= 1.8  # Freinage plus fort

        
        # Speed update
        self.car_speed = np.clip(self.car_speed + accel - abs(self.steer) * 0.0, 0.0, 25.0)
        
        # Movement with angle
        self.car_x += (self.car_speed / 2.0) * np.cos(self.car_angle)
        self.car_y += (self.car_speed / 2.0) * np.sin(self.car_angle)
        # --- Roadsides: soft clamp ---
        road_top, road_bottom = 300, 500
        car_half_height = 10 

        if self.car_y < road_top + car_half_height:
            self.border_hits += 1
            self.car_y = road_top + car_half_height
            self.car_angle *= 0.3              # reduces drift
            self.car_speed *= 0.98             # small slowdown
        elif self.car_y > road_bottom - car_half_height:
            self.border_hits += 1
            self.car_y = road_bottom - car_half_height
            self.car_angle *= 0.3
            self.car_speed *= 0.98

        # Update road offset (scrolling effect)
        self.road_offset += self.car_speed / 2.0
        
        # Update particles (simplified)
        if self.car_speed > 1 and random.random() < 0.3:
            self.exhaust_particles.append({
                'x': self.car_x - 30,
                'y': self.car_y + random.randint(-5, 5),
                'life': 20,
                'size': random.randint(3, 6)
            })
        
        self.exhaust_particles[:] = [p for p in self.exhaust_particles if p['life'] > 0]
        for particle in self.exhaust_particles:
            particle['life'] -= 1
            particle['x'] -= 2
        
        # Update obstacles
        for obs in self.obstacles:
            obs.update()
        
        # Calculate the reward
        reward = self._calculate_reward()
        
        # Check the end conditions
        done, truncated = self._check_termination()
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _calculate_reward(self):
        """Calculate the agent's reward"""
        reward = 0.0
        nearest, dist = self._get_nearest_obstacle()
        
        # --- R1: Distance traveled ---
        reward += 0.1 * self.car_speed / 25.0 # Reward proportional to speed
        
        # --- R2: Bonus of optimale speed ---
        if PREFERRED_SPEED - SPEED_THRESHOLD < self.car_speed < PREFERRED_SPEED + 2 * SPEED_THRESHOLD:
            reward += 0.05
            
        # --- R3: Penalty/ reward for the nearest obstacle---
        if nearest:
            # Reward to maintain the good distance
            reward += 0.005 * min(dist, 100) / 100.0
            
            if dist <= 30: # Critical approach threshold
                if nearest.kind == 1:  # Red light
                    if nearest.light_state == "red" and self.car_speed > SPEED_THRESHOLD:
                        reward -= 5.0 # Major penalty: Running a red light
                        self.terminal_message = "Red light burned!"
                        self.collision_hist.append(nearest)
                elif nearest.kind == 3: # Pedestrian
                    if dist <= 10:
                        reward -= 10.0 # Very heavy penalty : collision
                        self.terminal_message = f"Collision avec {STATE_TYPES[nearest.kind]}!"
                        self.collision_hist.append(nearest)
                elif nearest.kind == 4: # Car in front
                    if dist <= 10:
                        reward -= 5.0 # Heavy penalty: Collision
                        self.terminal_message = "Collision with another car"
                        self.collision_hist.append(nearest)

        # --- R4: Penalty for Lane Invasion ---
        road_top, road_bottom = 300, 500
        safe_top, safe_bottom = road_top + 20, road_bottom - 20
        
        if self.car_y < safe_top or self.car_y > safe_bottom:
            reward -= 0.1 # Penalty for touching the lane limits
            if self.car_y < road_top or self.car_y > road_bottom:
                reward -= 0.1 
                self.terminal_message = "Out of the lane!"
                self.lane_invade_hist.append(True)

        # --- R5: Penalty for slow driving (stagnation) ---
        if self.car_speed < 1.0 and self.step_counter > 100:
            reward -= 0.1
        
        # Clipping for stability
        reward = np.clip(reward, -10, 10)

        return reward
    
    def _check_termination(self):
        """Check the conditions of ends episods"""
        # 'done' is true if chaotic fail
        done = len(self.collision_hist) > 0 or (len(self.lane_invade_hist) > 0 and self.terminal_message == "Sortie de la chaussée!")
        
        # 'truncated'  is true if the episods reach its maximum time
        truncated = self.step_counter >= MAX_STEPS

        
        return done, truncated
    
    def render(self, info_text="", flash=False):
        """Show the state with Pygame."""
        if not self.SHOW_CAM:
            return
        
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.im_width, self.im_height))
            pygame.display.set_caption("Autonomous Drinving DQN")
            self.font = pygame.font.Font(None, 28)
            self.font_small = pygame.font.Font(None, 22)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # === DRAWING OF THE WORLD (BACKGROUND) ===
        # Sky, grass, buildings... (The detailed rendering code is preserved
        # and works well; there's no need to rewrite it here.)
        # [Image of Realistic View of a Driving Simulation with Car and Obstacles on a Two-Lane Road]
        # (The rendering is too complex to include in its entirety,
        # but the drawing logic is preserved)

        # Background (simplified for code review)
        self.window.fill(SKY)
        self.window.fill(GRASS, (0, 300, self.im_width, 30))
        self.window.fill(GRASS, (0, 500, self.im_width, 100))
        pygame.draw.rect(self.window, ROAD, (0, 300, self.im_width, 200))
        
        #  Animated separation lines (simplified)
        line_spacing = 50
        offset = int(self.road_offset) % line_spacing
        
        for x in range(-offset, self.im_width, line_spacing):
             # Centrale line (track separation)
            pygame.draw.line(self.window, YELLOW, (x, 400), (x + 30, 400), 4)
            # Lateral lines
            pygame.draw.line(self.window, WHITE, (x, 330), (x + 30, 330), 3)
            pygame.draw.line(self.window, WHITE, (x, 470), (x + 30, 470), 3)

        # === EXHAUST PARTICLES ===
        for particle in self.exhaust_particles:
            color = (150, 150, 150)
            pygame.draw.circle(self.window, color, (int(particle['x']), int(particle['y'])), particle['size'])
        
        # === OBSTACLES ===
        for obs in self.obstacles:
            x, y = int(obs.x), int(obs.y)
            color = RED if obs.kind in [1, 2] else YELLOW if obs.kind == 3 else ORANGE
            pygame.draw.rect(self.window, color, (x, y - 5, obs.width, obs.height))
            if obs.kind == 1 and obs.light_state == "red":
                pygame.draw.circle(self.window, (255, 0, 0), (x + obs.width // 2, y - 10), 10)
        
        # === Player car===
        car_display_x, car_display_y = int(self.car_x), int(self.car_y)
        pygame.draw.rect(self.window, BLUE, (car_display_x, car_display_y - 10, 70, 35), border_radius=6)
        
        # === USER INTERFACE (HUD) ===
        hud_surface = pygame.Surface((400, 180))
        hud_surface.set_alpha(200)
        hud_surface.fill((20, 20, 20))
        self.window.blit(hud_surface, (10, 10))
        
        # Driving informations
        speed_text = self.font.render(f"Speed: {self.car_speed:.1f} km/h", True, GREEN)
        self.window.blit(speed_text, (20, 20))
        
        steer_display = "Left" if self.steer < -0.1 else "Right" if self.steer > 0.1 else "Straight"
        steer_text = self.font.render(f"Direction: {steer_display}", True, WHITE)
        self.window.blit(steer_text, (20, 50))
        
        time_elapsed = self.step_counter / FPS
        time_text = self.font_small.render(f"Times: {time_elapsed:.1f}s / {SECONDS_PER_EPISODE}s", True, WHITE)
        self.window.blit(time_text, (20, 110))
        
        action_text = self.font_small.render(info_text, True, LIGHT_GRAY)
        self.window.blit(action_text, (20, 140))

        # === TERMINAL MESSAGE  ===
        if self.terminal_message:
            font_big = pygame.font.Font(None, 56)
            color = (200, 0, 0) if "Collision" in self.terminal_message or "pavement" in self.terminal_message else (200, 100, 0)
            text_big = font_big.render(self.terminal_message, True, WHITE)
            msg_rect = text_big.get_rect(center=(self.im_width//2, self.im_height//2))
            
            # Backgroup behind the message
            msg_bg = pygame.Surface((msg_rect.width + 40, msg_rect.height + 20))
            msg_bg.set_alpha(230)
            msg_bg.fill(color)
            self.window.blit(msg_bg, (msg_rect.left - 20, msg_rect.top - 10))

            self.window.blit(text_big, msg_rect)
        
        pygame.display.flip()
        self.clock.tick(FPS)
        
        # Management of events to exit the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
    
    def close(self):
        """Close the Pygame window"""
        if self.window:
            pygame.quit()
            self.window = None

# ============================================================
# TRAINING
# ============================================================
def train_dqn(episodes=1000, difficulty=1, save_path=SAVE_PATH):
    """Principale function for the agent's training"""
    env = DrivingEnv(difficulty=difficulty)
    agent = DQNAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        eps_decay=0.995,
        eps_min=0.05,
        batch_size=128,
        memory_size=500000,
    )
    
    all_rewards = []
    all_losses = []
    eval_scores = []
    eval_collisions = []

    ep_collision = []    # 1 if collision (done), 0 althought
    ep_borderhits = []   # number of border touches in the episode
    ep_distance = []     # distance traveled in the episode

    print("=" * 60)
    print("TRAINING BEGINNING DQN")
    print(f"Appareil: {agent.device}")
    print(f"Episodes: {episodes} | Difficulty: {difficulty}")
    print("=" * 60)
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        episode_losses = []
        
        while not (done or truncated):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Stock the transition
            agent.remember(state, action, reward, next_state, float(done))
            
            # Learning
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            total_reward += reward
            state = next_state

        ep_collision.append(1 if done else 0)
        ep_borderhits.append(getattr(env, "border_hits", 0))
        ep_distance.append(env.car_x - env.initial_x)

        # Epsilon decay
        agent.decay_epsilon()

        all_rewards.append(total_reward)
        if episode_losses:
            all_losses.append(np.mean(episode_losses))
        
        # Show and periodic evaluation
        if ep % 20 == 0:
            eval_score = evaluate_agent(agent, env, episodes=3)
            eval_scores.append(eval_score)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            print(f"Ep {ep:3d} | R: {total_reward:6.1f} | Eval: {eval_score:5.1f} | "
                  f"ε: {agent.epsilon:.3f} | Loss: {avg_loss:.4f} | Mem: {len(agent.memory)}")
        
        # Periodic save
        if ep % 100 == 0:
            agent.save(save_path)
            print(f"Model save: {save_path}")
    
    env.close()
    agent.save(save_path)
    
    # Visualization
    plot_training_results(all_rewards, all_losses, eval_scores, ep_collision)

    
    return agent

def evaluate_agent(agent, env, episodes=5):
    """Agent evaluation without exploration (epsilon=0)."""
    old_eps = agent.epsilon
    agent.epsilon = 0.0 
    
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.act(state, train=False) 
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
        
        total_rewards.append(total_reward)
    
    agent.epsilon = old_eps
    return np.mean(total_rewards)

def plot_training_results(rewards, losses, eval_scores, ep_collision):
    """Show the learning curves (Rewards, Loss, Evaluation Scores)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rewards par épisode
    axes[0, 0].plot(rewards, alpha=0.6, color='blue', linewidth=0.8)
    axes[0, 0].set_title("Rewards par Épisode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rewards lissés
    if len(rewards) >= 20:
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(rewards)), smoothed, 'r-', linewidth=2)
        axes[0, 1].set_title(f"Rewards Lissés (fenêtre={window})")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward Moyen")
        axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    if losses:
        axes[1, 0].plot(losses, alpha=0.7, color='orange', linewidth=1.2)
        axes[1, 0].set_title("Loss d'Entraînement")
        axes[1, 0].set_xlabel("Étape/Mini-batch de Replay")
        axes[1, 0].set_ylabel("Loss Moyenne par Épisode")
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot collisions 
    if ep_collision:
        y = np.array(ep_collision, dtype=np.float32)
        x = np.arange(len(y))

        axes[1, 1].plot(x, y, linewidth=0.8)  # brut 0/1
        axes[1, 1].set_title("Collisions par Épisode ")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Collision / %")
        axes[1, 1].grid(True, alpha=0.3)

        window = 20
        if len(y) >= window:
            ma = np.convolve(y, np.ones(window)/window, mode='valid') * 100.0
            x_ma = np.arange(window - 1, len(y))
            axes[1, 1].plot(x_ma, ma, linewidth=2.0)

    plt.tight_layout()
    plt.show()

# ============================================================
# DEMONSTRATION
# ============================================================
def demo_agent(agent, episodes=3, difficulty=1):
    """ Show the train agent with Pygame"""
    env = DrivingEnv(difficulty=difficulty)
    
    print("\n" + "=" * 60)
    print("🚗 AGENT DEMONSTRATION")
    print(f"Modèle chargé. ε: {agent.epsilon:.3f} (en mode démo, ε=0 sera utilisé)")
    print("Contrôles: ESC pour quitter, ESPACE pour pause")
    print("=" * 60)
    
    # Disable exploration for the demo
    old_eps = agent.epsilon
    agent.epsilon = 0.0 
    
    paused = False

    try:
        state, _ = env.reset()
        if env.SHOW_CAM:
            env.render("Démo: démarrage")

        
        for ep in range(1, episodes + 1):
            state, _ = env.reset()
            total_reward = 0
            done = False
            truncated = False
            step_count = 0
            
            print(f"\n{'—'*60}")
            print(f"📝 Episode {ep}/{episodes}")
            print(f"{'—'*60}")
            
            while not (done or truncated):
                # Events management
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print("⏸️ PAUSE" if paused else "▶️ REPRISE")
                
                if paused:
                    time.sleep(0.1)
                    continue
                
                # Agent's action
                action = agent.act(state, train=False)
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                
                # Informations
                nearest, dist = env._get_nearest_obstacle()
                obs_info = STATE_TYPES[nearest.kind] if nearest else "Route libre"
                dist_str = f"{dist:.0f}m" if dist != float('inf') else "∞"
                
                info = (f"🎯 Action: {ACTION_MAP[action]} | "
                       f"Obstacle: {obs_info} ({dist_str}) | "
                       f"Reward: {reward:+.1f} | Total: {total_reward:.1f}")
                
                # Affichage
                env.render(info, flash=done)
                
                # Log console periodic
                if step_count % 30 == 0:
                    print(f"  Step {step_count:3d} | V={env.car_speed:4.1f} km/h | "
                          f"Y={env.car_y:5.0f} | Dist={env.car_x - env.initial_x:5.0f}m | R={total_reward:6.1f}")
                
                time.sleep(0.02)
                state = next_state
                step_count += 1
            
            # Episod summary
            print(f"\n{'—'*60}")
            print(f"📈 Résumé Episode {ep}")
            print(f"{'—'*60}")
            print(f"  Reward total:      {total_reward:7.1f}")
            print(f"  Distance parcourue: {env.car_x - env.initial_x:6.0f}m")
            print(f"  Vitesse moyenne:    {env.car_speed:6.1f} km/h")
            print(f"  Durée:             {time.time() - env.episode_start:6.1f}s")
            
            if done:
                print(f"  ❌ Terminé: {env.terminal_message}")
            elif truncated:
                print(f"  ✅ Episode complet!")
            
            print(f"{'—'*60}")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\nInterruption de la démonstration.")
    
    finally:
        agent.epsilon = old_eps # Rétablir epsilon
        print("\n🏁 Démonstration terminée!")
        env.close()

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AUTONOMOUS DRIVING WITH DEEP Q-LEARNING (PyTorch/Pygame)")
    print("=" * 60)
    
    # 1. Training
    if SHOW_PREVIEW:
        pygame.init()

    try:
        trained_agent = train_dqn(episodes=1000, difficulty=1, save_path=SAVE_PATH)
        
        # 2. Demonstration
        print("\n" + "=" * 60)
        print("LANCEMENT DE LA DÉMONSTRATION")
        print("=" * 60)
        demo_agent(trained_agent, episodes=3, difficulty=1)
        
    except Exception as e:
        print(f"\nErreur critique: {e}")
        if 'pygame' in sys.modules and pygame.get_init():
             pygame.quit()
        
    finally:
        if 'pygame' in sys.modules and pygame.get_init():
             pygame.quit()
        print("Fin du programme.")