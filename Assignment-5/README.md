---
tags:
- SpaceInvadersNoFrameskip-v4
- deep-reinforcement-learning
- reinforcement-learning
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
      value: 900.0
      name: mean_reward
      verified: false
---

# PPO Agent playing SpaceInvadersNoFrameskip-v4

This is a trained model of a PPO agent playing SpaceInvadersNoFrameskip-v4 using CleanRL.

## Metrics
- **Mean Reward**: 900.0

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
- **Mean reward**: 900.00

## Hyperparameters

- Learning rate: 2.5e-4
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip coefficient: 0.1
- Value function coefficient: 0.5
- Entropy coefficient: 0.01
- Number of epochs: 4
- Minibatches: 4
