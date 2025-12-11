import os
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import numpy as np
import argparse

# Import the EXACT wrapper and agent used in training
from PPO import PPOAgent, CarRacingWrapper


def evaluate_and_record(
    model_path,
    env_name="CarRacing-v3",
    video_folder="video_out",
    num_episodes=3,
    img_stack=4,
    action_repeat=4
):
    """
    Loads a PPO model, runs it in the environment, and saves the video.
    Uses the EXACT SAME CarRacingWrapper from training.
    """
    print(f"Loading model from: {model_path}")

    # 1. Create the base environment with video recording enabled
    base_env = gym.make(env_name, render_mode="rgb_array", continuous=True)
    
    # 2. Wrap with RecordVideo
    os.makedirs(video_folder, exist_ok=True)
    video_env = RecordVideo(
        base_env,
        video_folder=video_folder,
        name_prefix="ppo_eval",
        episode_trigger=lambda x: True
    )

    # 3. Use the EXACT CarRacingWrapper from training, passing our video env
    env = CarRacingWrapper(
        env_name=env_name,
        img_stack=img_stack,
        action_repeat=action_repeat,
        env=video_env  # Pass the video-recording env
    )

    # 4. Initialize Agent and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        img_stack=img_stack,
        action_dim=3,
        device=device
    )
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    agent.load(model_path)

    # 5. Run the evaluation loop
    print("\nStarting Evaluation...")
    total_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        score = 0
        done = False
        truncated = False
        step = 0
        
        while not (done or truncated):
            # Deterministic=True makes the agent use its best predicted move (no random wiggling)
            action, _, _ = agent.select_action(state, deterministic=False)
            
            state, reward, done, truncated, info = env.step(action)
            score += reward
            step += 1
            
            # Die condition from wrapper
            if info.get('die', False):
                done = True

        print(f"Episode {ep+1}: Score = {score:.2f} (Steps: {step})")
        total_rewards.append(score)

    env.close()
    print(f"\nAverage Score: {np.mean(total_rewards):.2f}")
    print(f"Videos saved to: {video_folder}")


if __name__ == "__main__":
    # CONFIGURATION - Model path relative to this script's directory
    # Models are in the parent folder: ../models/
    MODEL_FILE = "../models/ppo_CarRacing-v3_best.pth"
    VIDEO_FOLDER = "../videos_eval/CarRacing-v3"

    # Check if model exists
    if not os.path.exists(MODEL_FILE):
        # Try alternative path
        alt_path = "../models/ppo_CarRacing-v3_t400000.pth"
        if os.path.exists(alt_path):
            MODEL_FILE = alt_path
        else:
            print(f"No model found at {MODEL_FILE}")
            print("Available models:")
            models_dir = "../models"
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    if f.endswith('.pth'):
                        print(f"  - {os.path.join(models_dir, f)}")
            exit()

    evaluate_and_record(
        model_path=MODEL_FILE,
        video_folder=VIDEO_FOLDER,
        num_episodes=10,
        img_stack=4,     # Must match training
        action_repeat=4  # Must match training
    )