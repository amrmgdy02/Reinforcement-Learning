import numpy as np
from grid_maze_env import GridMazeEnv
from typing import Tuple

class PolicyIterator:
    def __init__(self, env: GridMazeEnv, gamma: float = 0.5, theta: float = 1e-3, max_iterations: int = 500):
        """Initialize the Policy Iterator.

        Args:
            env (GridMazeEnv): OpenAI Gym environment.
            gamma (float, optional): Discount factor. Defaults to 0.5.
            theta (float, optional): Convergence threshold. Defaults to 1e-3.
            max_iterations (int, optional): Maximum number of policy iteration steps. Defaults to 500.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.env_size = env.size
        self.num_actions = len(env.action_to_direction)
        
        # Initialize policy and value function
        self.policy = np.ones((self.env_size, self.env_size, self.num_actions)) / self.num_actions
        self.value_function = np.zeros((self.env_size, self.env_size))

    def evaluate_policy(self) -> np.ndarray:
        """Evaluate the current policy until convergence.

        Returns:
            np.ndarray: The value function for the current policy.
        """
        V = np.zeros((self.env_size, self.env_size))
        
        # Initialize terminal state values
        # for x in range(self.env_size):
        #     for y in range(self.env_size):
        #         if self.env.is_terminal_state(np.array([x, y])):
        #             if (np.array([x, y]) == self.env._target_location).all():
        #                 V[x, y] = 10.0
        #             elif any((np.array([x, y]) == mine).all() for mine in self.env._mines):
        #                 V[x, y] = -10.0
                        
        while True:
            delta = 0
            V_new = np.copy(V)
            
            for x in range(self.env_size):
                for y in range(self.env_size):
                    if self.env.is_terminal_state(np.array([x, y])):
                        continue
                        
                    new_value = 0
                    current_policy_choice = self.policy[x, y]
                    
                    for action, p_a in enumerate(current_policy_choice):
                        action_value = 0
                        action_transition_model = self.env.transition_model[action]
                        
                        for actual_movement, p_s_new in action_transition_model.items():
                            direction = self.env.action_to_direction[actual_movement]
                            new_agent_location = np.clip(np.array([x, y]) + direction, 0, self.env_size - 1)
                            reward = self.env.calc_reward(new_agent_location)
                            action_value += p_s_new * (reward + self.gamma * V[new_agent_location[0], new_agent_location[1]])
                            
                        new_value += p_a * action_value
                    
                    V_new[x, y] = new_value
                    delta = max(delta, abs(V_new[x, y] - V[x, y]))
            
            V = V_new.copy()
            if delta < self.theta:
                break
                
        return V

    def improve_policy(self, V_values: np.ndarray) -> np.ndarray:
        """Improve the current policy based on the value function.

        Args:
            V_values (np.ndarray): Current value function.

        Returns:
            np.ndarray: The improved policy.
        """
        new_policy = self.policy.copy()
        
        for x in range(self.env_size):
            for y in range(self.env_size):
                if self.env.is_terminal_state(np.array([x, y])):
                    continue
                
                possible_states_info = self.env.get_possible_states(np.array([x, y]))
                
                if possible_states_info:
                    action_values = []
                    valid_actions = []
                    
                    for action, action_possible_states in possible_states_info:
                        action_value = 0
                        for prob, (new_x, new_y) in action_possible_states:
                            reward = self.env.calc_reward(np.array([new_x, new_y]))
                            action_value += prob * (reward + self.gamma * V_values[new_x, new_y])
                        action_values.append(action_value)
                        valid_actions.append(action)
                    
                    best_idx = np.argmax(action_values)
                    best_action = valid_actions[best_idx]
                    
                    new_policy[x, y] = np.zeros(self.num_actions)
                    new_policy[x, y][best_action] = 1.0

        return new_policy

    def iterate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the policy iteration algorithm.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The optimal policy and its value function.
        """
        old_policy = np.zeros_like(self.policy)
        num_iterations = 0
        for i in range(self.max_iterations):
            # Policy evaluation
            self.value_function = self.evaluate_policy()
            
            # Policy improvement
            self.policy = self.improve_policy(self.value_function)
            
            # Check convergence
            if np.allclose(self.policy, old_policy):
                num_iterations = i + 1
                break
                
            old_policy = self.policy.copy()
        
        return self.policy, self.value_function, num_iterations

    def visualize_policy(self):
        """Visualize the current policy using arrows."""
        print("VISUALIZING POLICY:")
        arrow_dict = {
            0: '→',  # Right
            1: '↑',  # Up
            2: '←',  # Left
            3: '↓'   # Down
        }
        
        for x in range(self.env_size):
            row_arrows = ''
            for y in range(self.env_size):
                best_action = np.argmax(self.policy[x, y])
                row_arrows += arrow_dict[best_action] + ' '
            print(row_arrows)
        print()

    def visualize_value_function(self):
        """Visualize the current value function."""
        print("VISUALIZING VALUE FUNCTION:")
        for x in range(self.env_size):
            row_values = ''
            for y in range(self.env_size):
                row_values += f"{self.value_function[x, y]:6.2f} "
            print(row_values)
        print()