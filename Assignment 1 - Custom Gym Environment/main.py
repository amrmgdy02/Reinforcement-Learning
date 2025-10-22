import numpy as np
from grid_maze_env import GridMazeEnv
from policy_iterator import PolicyIterator
from gymnasium.wrappers import RecordVideo

def get_agent_success_rate(env: GridMazeEnv, gamma: float, theta: float, num_episodes: int = 100) -> float:
        """ create agent for num_episodes environments and let them play the game.

        Args:
            num_episodes (int, optional): Number of episodes to run. Defaults to 100.
            env (GridMazeEnv): The environment to run the agent in.
            gamma (float): Discount factor for the Policy Iterator.
            theta (float): Convergence threshold for the Policy Iterator.
        """
        num_success = 0
        pi = PolicyIterator(env, gamma=gamma, theta=theta)
        optimal_policy, optimal_value_function, num_iterations = pi.iterate()
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = int(np.argmax(optimal_policy[state["agent"][0], state["agent"][1]]))
                state, reward, terminated, truncated, info_ = env.step(action)
                done = terminated or truncated
            # check if the agent reached the target
            if (state["agent"] == env._target_location).all():
                num_success += 1
                
        success_rate = num_success / num_episodes
        
        return success_rate

def get_diff_agents_success_rate(gamma: float, theta: float, num_episodes=100):
        """ create agent for num_episodes environments and let them play the game.

        Args:
            num_episodes (int, optional): Number of episodes to run. Defaults to 100.
        """
        num_success = 0
        success_rate = 0
        
        for _ in range(num_episodes):
            custom_env = GridMazeEnv(size=5, render_mode="rgb_array")
            pi = PolicyIterator(custom_env, gamma=gamma, theta=theta)
            optimal_policy, optimal_value_function, num_iterations = pi.iterate()
            
            state, _ = custom_env.reset()
            done = False
            
            while not done:
                action = int(np.argmax(optimal_policy[state["agent"][0], state["agent"][1]]))
                state, reward, terminated, truncated, info_ = custom_env.step(action)
                done = terminated or truncated
                if done:
                    if (state["agent"] == custom_env._target_location).all():
                        num_success += 1
                        
            success_rate = num_success / num_episodes
        
        return success_rate

def main():
    
    custom_env = GridMazeEnv(size=10, layout_seed=33, render_mode="rgb_array")
    
    # custom_env.save_env_as_image("imgs/grid_maze_env_1.png")
    
    pi = PolicyIterator(custom_env, gamma=0.7)
    optimal_policy, optimal_value_function, num_iterations = pi.iterate()
    
    print(f"Policy Iteration converged in {num_iterations} iterations.")
    pi.visualize_policy()
    pi.visualize_value_function()
    
    video_env = RecordVideo(custom_env, video_folder="videos", episode_trigger=lambda e: True, fps=2)
    
    state, _ = video_env.reset()
    done = False
    
    while not done:
        action = int(np.argmax(optimal_policy[state["agent"][0], state["agent"][1]]))
        state, reward, terminated, truncated, _ = video_env.step(action)
        done = terminated or truncated
        video_env.render()

    video_env.close()

if __name__ == "__main__":
    main()
    
    #success_rate = get_agent_success_rate(GridMazeEnv(size=5, layout_seed=33, render_mode="rgb_array"), gamma=0.99, theta=1e-3, num_episodes=100)
    #print(f"Agent Success Rate over 100 episodes: {success_rate * 100:.2f}%")
    
    # different_agents_success_rate = get_diff_agents_success_rate(gamma=0.99, theta=1e-3, num_episodes=100)
    # print(f"Different Agents Success Rate over 100 episodes: {different_agents_success_rate * 100:.2f}%")