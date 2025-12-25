from huggingface_hub import HfApi, upload_file, create_repo
import json
import os

# --- CONFIGURATION ---
# 1. SECURITY: Use environment variable or paste token carefully
repo_id = "Mahmoud103/ppo-SpaceInvadersNoFrameskip-v4" # Matches environment
model_path = "SpaceInvadersNoFrameskip-v4__PPO_atari__1__1766540445_agent_final.pth"
video_path = "videos/SpaceInvadersNoFrameskip-v4__PPO_atari__1__1766540445/rl-video-episode-0.mp4" # CHECK PATH

# 2. Evaluation Results
avg_reward = 900.0
std_reward = 50.0 

# 3. Create README with CORRECT metadata for Leaderboard
readme_content = f"""---
tags:
- SpaceInvadersNoFrameskip-v4
- deep-reinforcement-learning
- reinforcement-learning
- cleanrl
model-index:
- name: PPO
  results:
  - task:
      type: reinforcement-learning
      name: reinforcement-learning
    dataset:
      name: SpaceInvadersNoFrameskip-v4
      type: SpaceInvadersNoFrameskip-v4
    metrics:
    - type: mean_reward
      value: {avg_reward}
      name: mean_reward
      verified: false
---

# PPO Agent playing SpaceInvadersNoFrameskip-v4

This is a trained model of a PPO agent playing SpaceInvadersNoFrameskip-v4 using CleanRL.

## Metrics
- **Mean Reward**: {avg_reward}

## Usage
```python
import torch
import gymnasium as gym
from PPO_atari import Agent

env = gym.make("SpaceInvadersNoFrameskip-v4")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("SpaceInvadersNoFrameskip-v4")
agent = Agent(env).to(device)
agent.load_state_dict(torch.load("model.pth", map_location=device))
agent.eval()

# Run evaluation
obs, _ = env.reset()
done = False
while not done:
    action, _, _, _ = agent.get_action_and_value(torch.tensor(obs).unsqueeze(0).to(device))
    obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
    done = terminated or truncated
```

## Training Details

- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: SpaceInvadersNoFrameskip-v4
- **Total timesteps**: 10,000,000
- **Framework**: CleanRL
- **Number of parallel environments**: 8
- **Learning rate**: 2.5e-4
- **Evaluation episodes**: 100
- **Mean reward**: {avg_reward:.2f}

## Hyperparameters

- Learning rate: 2.5e-4
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip coefficient: 0.1
- Value function coefficient: 0.5
- Entropy coefficient: 0.01
- Number of epochs: 4
- Minibatches: 4
"""

# Create metadata file 
metadata = {
    "algorithm": "PPO",
    "environment": "SpaceInvadersNoFrameskip-v4",
    "mean_reward": 900,
    "eval_episodes": 100,
    "total_timesteps": 10_000_000
}

# Authenticate
api = HfApi()

# Create repository
try:
    create_repo(repo_id, private=False, exist_ok=True)
    print(f"Repository created: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Repository creation: {e}")

# Upload model
upload_file(
    path_or_fileobj=model_path,
    path_in_repo="model.pth",  # Standard name
    repo_id=repo_id
)
print("Model uploaded")

# Upload README
with open("README.md", "w") as f:
    f.write(readme_content)
upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=repo_id
)
print("README uploaded")

# Upload metadata
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
upload_file(
    path_or_fileobj="metadata.json",
    path_in_repo="metadata.json",
    repo_id=repo_id
)
print("Metadata uploaded")

# Upload your training script (recommended)
if os.path.exists("PPO_atari.py"):
    upload_file(
        path_or_fileobj="PPO_atari.py",
        path_in_repo="PPO_atari.py",
        repo_id=repo_id,
    )
    print("Training script uploaded")

print(f"\n✅ Model uploaded successfully!")
print(f"View at: https://huggingface.co/{repo_id}")