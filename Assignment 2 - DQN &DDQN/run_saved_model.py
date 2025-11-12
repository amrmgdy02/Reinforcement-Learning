from dqn import DDQNAgent
from main import evaluate, make_env  # you already have these
import torch
import os

# -----------------------------
# Environment setup
# -----------------------------
env_name = "CartPole-v1"  # change if needed
env = make_env(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# -----------------------------
# Load the saved model
# -----------------------------
model_dir = "./saved_models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, f"{env_name}_ddqn")  # base path

agent = DDQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.load(model_path)  # only base path; loads policy and target internally

# Evaluate the agent
mean_reward, all_rewards, episode_lengths = evaluate(agent, env_name, episodes=100)
print(f"\nAverage reward over 100 episodes: {mean_reward}")

from main import record_playback
record_playback(agent, env_name=env_name, video_dir=f"./videos_{env_name}_replay", episodes=2)
