import numpy as np
from grid_maze_env import GridMazeEnv
from policy_iterator import PolicyIterator
from gymnasium.wrappers import RecordVideo

def main():
    
    custom_env = GridMazeEnv(size=5, layout_seed=33, render_mode="rgb_array")
    
    pi = PolicyIterator(custom_env, gamma=0.7)
    optimal_policy, optimal_value_function = pi.iterate()
    
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