from typing import Optional
import numpy as np
import gymnasium as gym
import pygame

class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, size: int = 5, layout_seed: Optional[int]=None, num_mines: int = 2, cell_size: int = 100, render_mode="human"):

        super().__init__()
        self.size = size
        self.cell_size = cell_size
        self.render_mode = render_mode
        self.width = self.size * self.cell_size
        self.height = self.size * self.cell_size
        self._num_mines = num_mines
        
        self.transition_model = {
            0: {0: 0.7, 1: 0.15, 3: 0.15},  # Right
            1: {1: 0.7, 0: 0.15, 2: 0.15},  # Up
            2: {2: 0.7, 1: 0.15, 3: 0.15},  # Left
            3: {3: 0.7, 0: 0.15, 2: 0.15},  # Down
        }
        
        # self.transition_model = {
        #     0: {0: 1.0},  # Right
        #     1: {1: 1.0},  # Up
        #     2: {2: 1.0},  # Left
        #     3: {3: 1.0},  # Down
        # }
        
        assert size > 2, "Size must be greater than 2"
        
        if layout_seed is None:
            self._layout_seed = np.random.randint(0, 1_000_000)
        else:
            self._layout_seed = layout_seed
        
        # RNG just for layout initialization
        layout_rng = np.random.default_rng(seed=self._layout_seed)
        
        # TARGET
        self._target_location = layout_rng.integers(0, size, size=2, dtype=np.int32)
        
        # MINES
        self._mines = self._generate_mines(layout_rng)

        # AGENT
        self._agent_location = np.array([-1, -1], dtype=np.int32) # will be set randomly in reset() (each episode)
        
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "mines": gym.spaces.Box(0, size - 1, shape=(num_mines, 2), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self.action_to_direction = {
            0: np.array([0, 1]),   # Right: (row+0, col+1)
            1: np.array([-1, 0]),  # Up: (row-1, col+0)
            2: np.array([0, -1]),  # Left: (row+0, col-1)
            3: np.array([1, 0]),   # Down: (row+1, col+0)
        }
        
        # PyGame setup
        self.screen = None
        self.clock = None
        self.running = False
    
    
    def _generate_mines(self, rng):
        """Generate random mines avoiding target."""
        mines = []
        max_attempts = 100
        
        for _ in range(self._num_mines):
            for _ in range(max_attempts):
                mine = rng.integers(0, self.size, size=2, dtype=np.int32)
                
                # Check collisions
                if (mine == self._target_location).all():
                    continue
                if any((mine == m).all() for m in mines):
                    continue
                
                mines.append(mine)
                break
        
        return np.array(mines, dtype=np.int32)
    
    
    def _generate_agent_position(self):
        max_attempts = 1000
        for attempt in range(max_attempts):
            # Note: update only agent position (mines and target must be fixed to be solved with policy iteration, otherwise the agent will be solving a different MDP at each episode)
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
            if not any((self._agent_location == mine).all() for mine in self._mines) and not (self._agent_location == self._target_location).all():
                break
        else:
            raise RuntimeError(f"Could not place agent after {max_attempts} attempts. Too many mines?")
    
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "agent": self._agent_location,
            "mines": self._mines,
            "target": self._target_location,
        }


    def get_possible_actions(self, state: np.ndarray) -> list[int]:
        """Get possible actions from a given state.
        """
        
        possible_actions = []
        for action in range(self.action_space.n):
            direction = self.action_to_direction[action]
            new_state = state + direction
            if (0 <= new_state[0] < self.size) and (0 <= new_state[1] < self.size):
                possible_actions.append(action)
        
        return possible_actions

    def get_possible_states(self, state: np.ndarray) -> list[tuple[int, list[tuple[int, tuple[int, int]]]]]:
        """Get possible resulting states with their probabilities from all actions in a given state.

        Args:
            state (np.ndarray): Current state. Shape [2] (row, col).
        Returns:
            list[tuple[int, list[tuple[int, tuple[int, int]]]]]: List of possible actions with their possible resulting states (row, col) along with their probabilities.
        """
        
        possible_states = []
        for action in self.get_possible_actions(state):
        #for action in range(self.action_space.n):
            action_possible_states = []
            direction_1 = self.action_to_direction[action]
            new_state_1 = np.clip(state + direction_1, 0, self.size - 1)
            action_possible_states.append((0.7, (new_state_1[0], new_state_1[1])))
            
            direction_2 = self.action_to_direction[(action + 1) % 4]
            new_state_2 = np.clip(state + direction_2, 0, self.size - 1)
            action_possible_states.append((0.15, (new_state_2[0], new_state_2[1])))

            direction_3 = self.action_to_direction[(action - 1) % 4]
            new_state_3 = np.clip(state + direction_3, 0, self.size - 1)
            action_possible_states.append((0.15, (new_state_3[0], new_state_3[1])))

            possible_states.append((action, action_possible_states))

        return possible_states
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: observation for the initial state
        """
        super().reset(seed=seed)
        
        self._generate_agent_position()

        return self._get_obs(), {}
    
    
    def _get_stochastic_movement(self, action):
        """Apply stochasticity to the intended movement direction.

        Args:
            action: The intended action as an integer (0-3)

        Returns:
            int: The actual action after applying noise
        """
        #return action  # deterministic for now
        prob = self.np_random.random()
        if prob <= 0.7:
            return action  
        elif prob < 0.85:
            return (action + 1) % 4
        else:
            return (action - 1) % 4
    
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        if isinstance(action, np.ndarray):
            action = action.item()
        # Map the discrete action (0-3) to a movement direction
        actual_movement = self._get_stochastic_movement(action)
        direction = self.action_to_direction[actual_movement]

        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)
        
        reward = self.calc_reward(self._agent_location)
        terminated = (self._agent_location == self._target_location).all() or any((self._agent_location == mine).all() for mine in self._mines)

        truncated = False
        info = {}
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, info
    
    def calc_reward(self, agent_pos)-> float:
        """calculate the reward of being at a given state

        Args:
            agent_pos (_type_): _description_

        Returns:
            float: _description_
        """
        if (agent_pos == self._target_location).all():
            return 1.0
        elif any((agent_pos == mine).all() for mine in self._mines):
            return -1.0
        else:
            return -0.01
        
    def is_terminal_state(self, state: np.ndarray) -> bool:
        """Check if a given state is terminal.

        Args:
            state (np.ndarray): State to check. Shape [2] (row, col).

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        return (state == self._target_location).all() or any((state == mine).all() for mine in self._mines)
    
    
    def show_env_info(self):
        """Print environment information for debugging."""
        print(f"Environment Size: {self.size}x{self.size}")
        print(f"Target Location: {self._target_location}")
        print(f"Mines Locations: {self._mines}")
        
    
    def save_env_as_image(self, filename: str):
        """Save the current environment state as an image file.

        Args:
            filename (str): Path to save the image file.
        """
        rgb_array = self.render()
        pygame.image.save(pygame.surfarray.make_surface(np.transpose(rgb_array, (1, 0, 2))), filename)
        

    def render(self):
        """Render the environment to the screen or as an RGB array.

        Returns:
            np.ndarray or None: RGB array if render_mode is "rgb_array", else None
        """
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Grid Maze")
            else:  # rgb_array mode
                self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.running = True

        # Fill background
        self.screen.fill((255, 255, 255))  # white background

        # Draw grid lines
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.width, y))

        # Draw target
        goal_rect = pygame.Rect(self._target_location[1]*self.cell_size, self._target_location[0]*self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)  # green
        
        # Draw mines
        for mine in self._mines:
            mine_rect = pygame.Rect(mine[1]*self.cell_size, mine[0]*self.cell_size,
                                    self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), mine_rect)  # red
            
        # Draw agent as circle
        center_x = self._agent_location[1]*self.cell_size + self.cell_size//2
        center_y = self._agent_location[0]*self.cell_size + self.cell_size//2
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, (0, 0, 255), (center_x, center_y), radius)  # blue
        
        # Handle render modes
        if self.render_mode == "human":
            pygame.display.get_surface().blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            # pygame.time.delay(100)
        elif self.render_mode == "rgb_array":
            # return RGB array for video recording
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))
        
        
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.running = False