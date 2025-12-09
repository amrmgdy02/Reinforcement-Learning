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
SAC_CONFIGS = {
    "LunarLander-v3": {
        "total_timesteps": 750_000,
        "start_timesteps": 10_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "lr": 3e-4,
        "buffer_capacity": 500_000,
        "eval_episodes": 5,
        "eval_max_steps": 1000,
        "continuous_action_space": True,
        "use_cnn": False,  # MLP for LunarLander
    },

    # "CarRacing-v3": {
    #     "total_timesteps": 1_000_000,
    #     "start_timesteps": 20_000,
    #     "batch_size": 128,  # Reduced for memory efficiency with images
    #     "gamma": 0.99,
    #     "tau": 0.005,
    #     "alpha": 0.1,
    #     "lr": 3e-4,
    #     "buffer_capacity": 100_000,  # Reduced due to image storage
    #     "eval_episodes": 5,
    #     "eval_max_steps": 2000,
    #     "continuous_action_space": True,
    #     "use_cnn": True,  # CNN for CarRacing
    #     "frame_skip": 4,  # Skip frames to speed up training
    #     "grayscale": False,  # Keep RGB for better visual info
    # }
    "CarRacing-v3": {
        "total_timesteps": 30000,
        "start_timesteps": 20_000,
        "batch_size": 128,  # Reduced for memory efficiency with images
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.1,
        "lr": 3e-4,
        "buffer_capacity": 100_000,  # Reduced due to image storage
        "eval_episodes": 5,
        "eval_max_steps": 2000,
        "continuous_action_space": True,
        "use_cnn": True,  # CNN for CarRacing
        "frame_skip": 4,  # Skip frames to speed up training
        "grayscale": False,  # Keep RGB for better visual info
    }
}


# =========================================================
# Helper Functions
# =========================================================
def flatten_state(s, use_cnn=False):
    """
    Flatten state for storage in replay buffer.
    For CNN: keep image structure but flatten for buffer storage
    For MLP: flatten completely
    """
    s = np.array(s, dtype=np.float32)
    
    if use_cnn and s.ndim == 3:  # Image: (H, W, C)
        # Normalize to [0, 1]
        s = s / 255.0
        # Convert from (H, W, C) to (C, H, W) for PyTorch
        s = np.transpose(s, (2, 0, 1))
        # Flatten for buffer storage
        return s.reshape(-1)
    elif s.ndim == 3:  # Image but not using CNN
        s = s / 255.0 
        return s.reshape(-1)
    else:  # Vector state
        return s.reshape(-1)


def preprocess_frame(frame, use_cnn=False):
    """
    Preprocess a single frame for the agent.
    For CNN: resize and normalize
    For MLP: flatten
    """
    frame = np.array(frame, dtype=np.float32)
    
    if use_cnn and frame.ndim == 3:
        # CarRacing default is 96x96x3, normalize to [0,1]
        frame = frame / 255.0
        # Convert to (C, H, W) format
        frame = np.transpose(frame, (2, 0, 1))
        return frame
    else:
        # For MLP or vector states
        if frame.ndim == 3:
            frame = frame / 255.0
        return frame.reshape(-1)


# =========================================================
# TRAINING FUNCTION
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
    use_cnn = cfg.get("use_cnn", False)
    frame_skip = cfg.get("frame_skip", 1)

    # ---------------------------
    # Device setup
    # ---------------------------
    device = get_device()
    print(f"\nUsing device: {device}")
    print(f"Training SAC on: {env_name}")
    print(f"Using {'CNN' if use_cnn else 'MLP'} architecture\n")

    # ---------------------------
    # W&B initialization
    # ---------------------------
    wandb.login()
    run_name = f"SAC_{env_name}_{'CNN' if use_cnn else 'MLP'}"

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
    
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if use_cnn:
        # For CNN, we need the image shape
        input_channels = sample_obs.shape[2] if sample_obs.ndim == 3 else 3
        state_dim = None  # Not used for CNN
        print(f"Image shape: {sample_obs.shape}")
        print(f"Input channels: {input_channels}, Action dim: {action_dim}, Max action: {max_action}\n")
    else:
        # For MLP, we need the flattened state dimension
        state = flatten_state(sample_obs, use_cnn=False)
        state_dim = state.shape[0]
        input_channels = None
        print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action}\n")

    # ---------------------------
    # Agent + Buffer
    # ---------------------------
    agent = SACAgent(
        state_dim=state_dim if not use_cnn else 0,
        action_dim=action_dim,
        max_action=max_action,
        lr=lr,
        gamma=gamma,
        tau=tau,
        alpha=alpha,
        device=device,
        use_cnn=use_cnn,
        input_channels=input_channels if use_cnn else 3
    )
    replay_buffer = ReplayBuffer(buffer_capacity)

    # ---------------------------
    # TRAINING LOOP
    # ---------------------------
    episode_rewards = []
    episode_reward = 0
    episode_duration = 0
    steps_since_log = 0
    frame_count = 0

    state = preprocess_frame(env.reset()[0], use_cnn)

    for t in range(total_timesteps):

        # Action selection
        if len(replay_buffer) < start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        # Step with frame skip
        total_reward = 0
        done = False
        truncated = False
        
        for _ in range(frame_skip):
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            frame_count += 1
            if done or truncated:
                break
        
        next_state = preprocess_frame(next_state, use_cnn)

        # Store transition (flatten for buffer storage)
        state_flat = state.reshape(-1) if use_cnn else state
        next_state_flat = next_state.reshape(-1) if use_cnn else next_state
        replay_buffer.push(Transition(state_flat, action, total_reward, next_state_flat, done or truncated))

        state = next_state
        episode_reward += total_reward
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
                "timestep": t,
                "frames": frame_count
            })

            episode_rewards.append(episode_reward)
            episode_reward = 0
            episode_duration = 0
            state = preprocess_frame(env.reset()[0], use_cnn)

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
        state = preprocess_frame(eval_env.reset()[0], use_cnn)
        done = False
        ep_reward = 0
        steps = 0

        while not done and steps < eval_max_steps:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, info = eval_env.step(action)
            state = preprocess_frame(next_state, use_cnn)
            ep_reward += reward
            steps += 1
            done = done or truncated

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
    
    # Get config to determine if CNN is needed
    if env_name not in SAC_CONFIGS:
        raise ValueError(f"No SAC hyperparameters found for {env_name}")
    
    cfg = SAC_CONFIGS[env_name]
    use_cnn = cfg.get("use_cnn", False)

    env_video_dir = os.path.join(eval_video_dir, env_name)
    os.makedirs(env_video_dir, exist_ok=True)
    env = gym.make(env_name, continuous=True, render_mode="rgb_array")
    env = RecordVideo(env, env_video_dir, episode_trigger=lambda e: True)

    sample_obs, _ = env.reset()
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    if use_cnn:
        input_channels = sample_obs.shape[2] if sample_obs.ndim == 3 else 3
        state_dim = 0
    else:
        state_dim = flatten_state(sample_obs, use_cnn=False).shape[0]
        input_channels = 3

    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        use_cnn=use_cnn,
        input_channels=input_channels
    )
    agent.load(model_path)

    for ep in range(eval_episodes):
        state = preprocess_frame(env.reset()[0], use_cnn)
        done = False
        ep_reward = 0
        steps = 0

        while not done and steps < eval_max_steps:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, info = env.step(action)
            state = preprocess_frame(next_state, use_cnn)
            ep_reward += reward
            steps += 1
            done = done or truncated

        print(f"[Eval] Episode {ep+1} reward = {ep_reward:.2f}")

    env.close()
    print(f"Videos saved to: {env_video_dir}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # Train on LunarLander (uses MLP)
    #train_sac_with_eval("LunarLander-v3")

    # Train on CarRacing (uses CNN)
    train_sac_with_eval("CarRacing-v3")

    # Evaluate only (LunarLander)
    #eval_sac_without_training("LunarLander-v3", "models/LunarLander-v3/sac_model")
    
    # Evaluate only (CarRacing)
    # eval_sac_without_training("CarRacing-v3", "models/CarRacing-v3/sac_model")