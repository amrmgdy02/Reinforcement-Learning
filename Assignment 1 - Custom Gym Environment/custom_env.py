from typing import Optional
import numpy as np
import gymnasium as gym

class GridMazeEnv(gym.Env):

    def __init__(self, size: int = 5, mines = np.array([(1, 1), (2, 3)])):
        
        super().__init__()
        self.size = size
        
        assert size > 2, "Size must be greater than 2"
        
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32) # will be set randomly in reset()
        self._target_location = np.array([-1, -1], dtype=np.int32) # will be set randomly in reset()
        self._mines = mines

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "mines": gym.spaces.Box(0, size - 1, shape=(2, 2), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),   # Move right (positive x)
            1: np.array([0, 1]),   # Move up (positive y)
            2: np.array([-1, 0]),  # Move left (negative x)
            3: np.array([0, -1]),  # Move down (negative y)
        }
    
    
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
    
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: observation for the initial state
        """
        super().reset(seed=seed)
        
        while True:
            # Note: update only agent position (mines and target must be fixed to be solved with policy iteration, otherwise the agent will be solving a different MDP at each episode)
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=np.int32)
            if not any((self._agent_location == mine).all() for mine in self._mines) and not (self._agent_location == self._target_location).all():
                break

        return self._get_obs()
    
    
    def _get_stochastic_movement(self, action):
        """Apply stochasticity to the intended movement direction.

        Args:
            action: The intended action as an integer (0-3)

        Returns:
            int: The actual action after applying noise
        """
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
        # Map the discrete action (0-3) to a movement direction
        chosen_action = self._get_stochastic_movement(action)
        direction = self._action_to_direction[chosen_action]
        
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Check for termination conditions
        if (self._agent_location == self._target_location).all():
            reward = 10.0
            terminated = True
        
        elif any((self._agent_location == mine).all() for mine in self._mines):
            reward = -10.0
            terminated = True
            
        else:
            reward = -1.0
            terminated = False

        truncated = False
        info = {}
        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, info