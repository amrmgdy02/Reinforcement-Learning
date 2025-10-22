import numpy as np
from custom_env import GridMazeEnv
from gymnasium.wrappers import RecordVideo

def policy_evaluation(env: GridMazeEnv, policy: np.ndarray, gamma: float = 1.0, theta: float = 1e-3) -> np.ndarray:
    """Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        env (GridMazeEnv): OpenAI Gym environment.
        policy (np.ndarray): Policy to evaluate. Shape [num_states, num_actions].
        gamma (float): Gamma discount factor.
        theta (float): A threshold of value function change for stopping criteria.

    Returns:
        np.ndarray: The value function for the given policy. Shape [num_states].
    """
    env_size = env.size
    V = np.zeros((env_size, env_size))
    
    #Initialize terminal state values
    # for x in range(env_size):
    #     for y in range(env_size):
    #         if env.is_terminal_state([x, y]):
    #             if (np.array([x, y]) == env._target_location).all():
    #                 V[x, y] = 10.0
    #             elif any((np.array([x, y]) == mine).all() for mine in env._mines):
    #                 V[x, y] = -10.0
                    
    while True:
        delta = 0
        V_new = np.copy(V)
        
        for x in range(env_size):
            for y in range(env_size):
                if env.is_terminal_state([x, y]):
                    continue
                    
                new_value = 0
                current_policy_choice = policy[x, y]
                
                for action, p_a in enumerate(current_policy_choice):
                    action_value = 0
                    action_transition_model = env.transition_model[action]
                    
                    for actual_movement, p_s_new in action_transition_model.items():
                        direction = env.action_to_direction[actual_movement]
                        new_agent_location = np.clip(np.array([x, y]) + direction, 0, env.size - 1)
                        reward = env.calc_reward(new_agent_location)
                        action_value += p_s_new * (reward + gamma * V[new_agent_location[0], new_agent_location[1]])
                        
                    new_value += p_a * action_value
                
                V_new[x, y] = new_value
                delta = max(delta, abs(V_new[x, y] - V[x, y]))
        
        # Update V and check for convergence
        V = V_new.copy()
        #break
        if delta < theta:
            break
            
    return V


def policy_improvement(env: GridMazeEnv, policy: np.ndarray, V_values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Improve a policy given a value function.

    Args:
        env (GridMazeEnv): OpenAI Gym environment.
        policy (np.ndarray): The current policy. Shape [num_states, num_actions].
        V (np.ndarray): The value function. Shape [num_states].
        gamma (float): Gamma discount factor.

    Returns:
        np.ndarray: The improved policy. Shape [num_states, num_actions].
    """
    env_size = env.size
    policy = policy.copy()
    
    for x in range(env_size):
        for y in range(env_size):
            # Skip terminal states
            if env.is_terminal_state([x, y]):
                continue
            
            # Get possible actions and their resulting states for current position
            possible_states_info = env.get_possible_states([x, y])
            
            if possible_states_info:
                action_values = []
                valid_actions = []
                
                # Evaluate each possible action
                for action, action_possible_states in possible_states_info:
                    action_value = 0
                    for prob, (new_x, new_y) in action_possible_states:
                        reward = env.calc_reward([new_x, new_y])
                        action_value += prob * (reward + gamma * V_values[new_x, new_y])
                    action_values.append(action_value)
                    valid_actions.append(action)
                
                # Choose best among valid actions
                best_idx = np.argmax(action_values)
                best_action = valid_actions[best_idx]
                
                policy[x, y] = np.zeros(len(env.action_to_direction))
                policy[x, y][best_action] = 1.0

    return policy


def policy_iteration(env: GridMazeEnv, gamma: float = 0.5, theta: float = 1e-3, max_iterations: int = 500):
    """Run policy iteration algorithm.

    Args:
        env (GridMazeEnv): OpenAI Gym environment.
        gamma (float): Gamma discount factor.
        theta (float): A threshold of value function change for stopping criteria.
        max_iterations (int): Maximum number of iterations to run.

    Returns:
        (np.ndarray, np.ndarray): The optimal policy and its value function.
    """
    env_size = env.size
    num_actions = len(env.action_to_direction)
  
    # Initialize a random policy
    new_policy = np.ones((env_size, env_size, num_actions)) / num_actions
    old_policy = np.zeros_like(new_policy)
    
    for i in range(max_iterations):
        V = policy_evaluation(env, new_policy, gamma, theta)
        new_policy = policy_improvement(env, new_policy, V, gamma)
        if np.allclose(new_policy, old_policy):
            break

        old_policy = new_policy.copy()

    return new_policy, V


def visualize_policy(policy: np.ndarray):
    """Visualize the policy using arrows.

    Args:
        policy (np.ndarray): The policy to visualize. Shape [num_states, num_actions].
    """
    arrow_dict = {
        0: '→',  # Right
        1: '↑',  # Up
        2: '←',  # Left
        3: '↓'   # Down
    }
    env_size = policy.shape[0]
    for x in range(env_size):
        row_arrows = ''
        for y in range(env_size):
            best_action = np.argmax(policy[x, y])
            row_arrows += arrow_dict[best_action] + ' '
        print(row_arrows)


################################ TEST TIME ###################################
custom_env = GridMazeEnv(size=3, layout_seed=42, render_mode="rgb_array")
optimal_policy, optimal_value_function = policy_iteration(custom_env, gamma=0.5)
visualize_policy(optimal_policy)
print("Optimal Value Function:", optimal_value_function)

video_env = RecordVideo(custom_env, video_folder="videos", episode_trigger=lambda e: True, fps=2)

state, _ = video_env.reset()
done = False
while not done:
    print("Current State:", state["agent"])
    action = int(np.argmax(optimal_policy[state["agent"][0], state["agent"][1]]))
    print("Taking Action:", action)
    state, reward, terminated, truncated, _ = video_env.step(action)
    done = terminated or truncated

    video_env.render()

video_env.close()