# =========================================================
# SAC Training Script (Supports Multi-Environment Configs)
# =========================================================

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import wandb

# ---------------------------
# Import your SAC implementation
# ---------------------------
from SAC import SACAgent, ReplayBuffer, Transition, get_device

# =========================================================
# ENVIRONMENT-SPECIFIC SAC CONFIGURATIONS
# =========================================================
# =========================================================
SAC_CONFIGS = {
    "LunarLander-v3": {
        "total_timesteps": 750_000,   # Increased for robust convergence
        "start_timesteps": 10_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,                 # Keep fixed, or set to 'auto' if supported
        "lr": 3e-4,
        "buffer_capacity": 500_000,   # Slightly increased for better data diversity
        "eval_episodes": 5,
        "eval_max_steps": 1000,
        "continuous_action_space": True,
    },

    "CarRacing-v3": {
        "total_timesteps": 500_000,
        "start_timesteps": 20_000,
        "batch_size": 512,
        "gamma": 0.99,
        "tau": 0.01,
        "alpha": 0.2,  
        "lr": 1e-4,
        "buffer_capacity": 1_000_000,
        "eval_episodes": 5,
        "eval_max_steps": 2000,
        "continuous_action_space": True,
    }
}


# =========================================================
# Helper
# =========================================================
def flatten_state(s):
    """Flatten state to 1D array for MLP networks"""
    return np.array(s, dtype=np.float32).reshape(-1)


# =========================================================
# TRAINING FUNCTION (NOW USES CONFIGS AUTOMATICALLY)
# =========================================================
def train_sac_with_eval(
    env_name,
    wandb_project="SAC-Box2D-Training",
    save_models=True,
    models_dir="models",
    eval_video_dir="videos_eval"
):

    # ---------------------------
    # Load environment hyperparameters
    # ---------------------------
    if env_name not in SAC_CONFIGS:
        raise ValueError(f"No SAC hyperparameters found for {env_name}")

    cfg = SAC_CONFIGS[env_name]
    total_timesteps = cfg["total_timesteps"]
    start_timesteps = cfg["start_timesteps"]
    batch_size = cfg["batch_size"]
    gamma = cfg["gamma"]
    tau = cfg["tau"]
    alpha = cfg["alpha"]
    lr = cfg["lr"]
    buffer_capacity = cfg["buffer_capacity"]
    eval_episodes = cfg["eval_episodes"]
    eval_max_steps = cfg["eval_max_steps"]
    continuous_action_space = cfg["continuous_action_space"]

    # ---------------------------
    # Device setup
    # ---------------------------
    device = get_device()
    print(f"\nUsing device: {device}")
    print(f"Training SAC on: {env_name}\n")

    # ---------------------------
    # W&B initialization
    # ---------------------------
    wandb.login()
    run_name = f"SAC_{env_name}"

    run = wandb.init(
        project=wandb_project,
        name=run_name,
        config=cfg
    )

    # ---------------------------
    # Environment setup
    # ---------------------------
    env = gym.make(env_name, continuous=continuous_action_space)
    sample_obs, _ = env.reset()
    state = flatten_state(sample_obs)
    state_dim = state.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}\n")

    # ---------------------------
    # Agent + Buffer
    # ---------------------------
    agent = SACAgent(state_dim, action_dim, max_action, lr, gamma, tau, alpha, device)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    episode_rewards = []
    episode_reward = 0
    episode_duration = 0
    steps_since_log = 0

    state = flatten_state(env.reset()[0])

    for t in range(total_timesteps):

        # Action selection
        if len(replay_buffer) < start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        # Step
        next_state, reward, done, truncated, info = env.step(action)
        next_state = flatten_state(next_state)

        # Store transition
        replay_buffer.push(Transition(state, action, reward, next_state, done or truncated))

        state = next_state
        episode_reward += reward
        episode_duration += 1

        # Train SAC
        if len(replay_buffer) >= batch_size and len(replay_buffer) >= start_timesteps:
            critic_loss, actor_loss = agent.train(replay_buffer, batch_size)

        # Episode end
        if done or truncated:
            print(f"Episode {len(episode_rewards)+1} | Reward = {episode_reward:.2f} | Steps = {episode_duration}")
            wandb.log({
                "episode_reward": episode_reward,
                "episode_duration": episode_duration,
                "timestep": t
            })

            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_duration = 0
            state = flatten_state(env.reset()[0])

        # Log avg reward
        steps_since_log += 1
        if steps_since_log >= 10_000:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"[{t}] Avg reward (last 10): {avg_reward:.2f}")
            wandb.log({"avg_reward_last_10": avg_reward, "timestep": t})
            steps_since_log = 0

    # ---------------------------
    # SAVE MODELS
    # ---------------------------
    if save_models:
        os.makedirs(os.path.join(models_dir, env_name), exist_ok=True)
        model_path = os.path.join(models_dir, env_name, "sac_model")
        agent.save(model_path)
        print(f"SAC model saved to: {model_path}")

    wandb.finish()

    # ---------------------------
    # EVALUATION + VIDEO RECORDING
    # ---------------------------
    print("\nStarting evaluation...")

    env_video_dir = os.path.join(eval_video_dir, env_name)
    os.makedirs(env_video_dir, exist_ok=True)
    eval_env = gym.make(env_name, continuous=True, render_mode="rgb_array")
    eval_env = RecordVideo(eval_env, env_video_dir, episode_trigger=lambda e: True)

    for ep in range(eval_episodes):
        state = flatten_state(eval_env.reset()[0])
        done = False
        ep_reward = 0
        steps = 0

        while not done and steps < eval_max_steps:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, info = eval_env.step(action)
            state = flatten_state(next_state)
            ep_reward += reward
            steps += 1

        print(f"[Eval] Episode {ep+1} reward = {ep_reward:.2f}")

    eval_env.close()
    print(f"Evaluation videos saved to: {env_video_dir}\n")


# =========================================================
# EVALUATE ONLY (NO TRAINING)
# =========================================================
def eval_sac_without_training(
    env_name,
    model_path,
    eval_episodes=5,
    eval_max_steps=1000,
    eval_video_dir="videos_eval"
):
    device = get_device()

    env_video_dir = os.path.join(eval_video_dir, env_name)
    os.makedirs(env_video_dir, exist_ok=True)
    env = gym.make(env_name, continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, env_video_dir, episode_trigger=lambda e: True)

    sample_obs, _ = env.reset()
    state_dim = flatten_state(sample_obs).shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action, device=device)
    agent.load(model_path)

    for ep in range(eval_episodes):
        state = flatten_state(env.reset()[0])
        done = False
        ep_reward = 0
        steps = 0

        while not done and steps < eval_max_steps:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, info = env.step(action)
            state = flatten_state(next_state)
            ep_reward += reward
            steps += 1

        print(f"[Eval] Episode {ep+1} reward = {ep_reward:.2f}")

    env.close()
    print(f"Videos saved to: {env_video_dir}")



# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # Train on LunarLander
    #train_sac_with_eval("LunarLander-v3")

    # Train on CarRacing
    # train_sac_with_eval("CarRacing-v3")

    # Evaluate only (LunarLander)
    eval_sac_without_training("LunarLander-v3", "models/LunarLander-v3/sac_model")

