import numpy as np
from custom_env import GridMazeEnv

def policy_evaluation(env: GridMazeEnv, policy: np.ndarray, gamma: float = 1.0, theta: float = 1e-6) -> np.ndarray:
    """Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        env (GridMazeEnv): OpenAI Gym environment.
        policy (np.ndarray): Policy to evaluate. Shape [num_states, num_actions].
        gamma (float): Gamma discount factor.
        theta (float): A threshold of value function change for stopping criteria.

    Returns:
        np.ndarray: The value function for the given policy. Shape [num_states].
    """
    