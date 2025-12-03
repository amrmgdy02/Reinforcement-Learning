import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import os

from SAC import SACAgent, ReplayBuffer, Transition


# ==========================================
# ENVIRONMENT CONFIG: Hyperparameters per env
# ==========================================
ENV_CONFIG = {
    "CartPole-v1": {
    "episodes": 500,
    "max_steps": 500,
    "batch_size": 128,
    "replay_size": 100_000,
    "lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005, #tau for soft update it , inc tau makes the update faster, dec tau makes it slower
    "alpha": 0.05, #entropy coefficient (H) to balance exploration and exploitation
},

    "Acrobot-v1": {
        "episodes": 600,
        "max_steps": 500,
        "batch_size": 256,
        "replay_size": 200_000,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
    },

    "MountainCar-v0": {
        "episodes": 900,
        "max_steps": 500,
        "batch_size": 256,
        "replay_size": 300_000,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.5,
    },

    "Pendulum-v1": {
        "episodes": 400,
        "max_steps": 200,
        "batch_size": 256,
        "replay_size": 300_000,
        "lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.05,
    },
}


# ==========================================
# Helper: Create agent based on action space
# ==========================================
def init_agent(env, config):
    obs_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.Box):
        return SACAgent(
            state_dim=obs_dim,
            action_dim=env.action_space.shape[0],
            max_action=float(env.action_space.high[0]),
            lr=config["lr"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            action_type="continuous"
        )

    elif isinstance(env.action_space, gym.spaces.Discrete):
        return SACAgent(
            state_dim=obs_dim,
            action_dim=env.action_space.n,
            max_action=1.0,
            lr=config["lr"],
            gamma=config["gamma"],
            tau=config["tau"],
            alpha=config["alpha"],
            action_type="discrete"
        )

    raise ValueError("Unsupported action space type!")


# ==========================================
# TRAIN FUNCTION
# ==========================================
def train(env_id):

    cfg = ENV_CONFIG[env_id]

    # Create model directory per environment
    model_dir = os.path.join("models", env_id)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sac_model")

    env = gym.make(env_id)
    agent = init_agent(env, cfg)
    buffer = ReplayBuffer(cfg["replay_size"])

    print(f"\n=== TRAINING {env_id} ({agent.action_type.upper()}) ===")

    for ep in range(cfg["episodes"]):
        state, _ = env.reset()
        ep_reward = 0
        c_loss_sum = 0
        a_loss_sum = 0
        updates = 0

        for step in range(cfg["max_steps"]):
            action = agent.select_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            buffer.push(Transition(state, action, reward, next_state, done))
            state = next_state
            ep_reward += reward

            if len(buffer) > cfg["batch_size"]:
                c_loss, a_loss = agent.train(buffer, cfg["batch_size"])
                c_loss_sum += c_loss
                a_loss_sum += a_loss
                updates += 1

            if done:
                break

        avg_c = c_loss_sum / updates if updates > 0 else 0
        avg_a = a_loss_sum / updates if updates > 0 else 0

        print(f"[{env_id}] Ep {ep+1}/{cfg['episodes']} | "
              f"Reward: {ep_reward:.2f} | C Loss: {avg_c:.4f} | A Loss: {avg_a:.4f}")

    # Save model into its environment folder
    agent.save(model_path)
    print(f"✅ Model saved to: {model_path}")

    env.close()


# ==========================================
# TEST FUNCTION
# ==========================================
def test(env_id):

    cfg = ENV_CONFIG[env_id]
    print(f"\n=== TESTING {env_id} ===")

    # Create video directory per environment
    video_dir = os.path.join("videos", env_id)
    os.makedirs(video_dir, exist_ok=True)

    model_path = os.path.join("models", env_id, "sac_model")

    env = gym.make(env_id, render_mode="rgb_array")
    env = RecordVideo(env, video_dir, name_prefix="test", episode_trigger=lambda e: e < 10)

    agent = init_agent(env, cfg)
    agent.load(model_path)

    rewards = []

    for ep in range(10):
        state, _ = env.reset(seed=ep)
        total = 0

        for step in range(cfg["max_steps"]):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, term, trunc, _ = env.step(action)

            state = next_state
            total += reward

            if term or trunc:
                break

        rewards.append(total)
        print(f"[{env_id}] Test Episode {ep+1}: Reward = {total:.2f}")

    print(f"\nAverage Test Reward for {env_id}: {np.mean(rewards):.2f}")
    env.close()


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    ENVIRONMENTS = [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "Pendulum-v1"
    ]

    for env_id in ENVIRONMENTS:
        #train(env_id)
        test(env_id)

    print("\n✔ All environments trained and tested successfully.")
