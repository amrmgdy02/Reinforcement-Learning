# =========================================================
# PPO Training Script for CarRacing-v3 (Optimized)
# =========================================================

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import wandb
from datetime import datetime
import argparse

from PPO import PPOAgent, CarRacingWrapper, get_device

# =========================================================
# PPO CONFIGURATION (OPTIMIZED)
# =========================================================
PPO_CONFIG = {
    "CarRacing-v3": {
        "total_timesteps": 500_000,    # User limit
        "max_ep_len": 1000,
        "start_timesteps": 2000,       # Lower this: Don't waste 10k steps on random actions in a short run
        "img_stack": 4,                # Standard (detects speed/direction)
        "action_repeat": 4,            # KEY CHANGE: 8 is too laggy; 4 gives better control for corners
        "buffer_capacity": 2048,       # KEY CHANGE: Use powers of 2 (2048) to align with batch_size
        "lr": 3e-4,                    # KEY CHANGE: 1e-3 causes instability. 3e-4 (0.0003) is the "Gold Standard"
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,             # KEY CHANGE: 0.2 allows faster updates than 0.1 without collapsing
        "ppo_epochs": 10,
        "batch_size": 64,              # KEY CHANGE: Smaller batches (64 vs 128) often generalize better for CNNs
        "max_grad_norm": 0.5,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,          # Keeps exploration active
        "log_interval": 10,
        "save_interval": 50_000,       # Save every 10%
        "eval_interval": 50_000,
        "eval_episodes": 5,
    }
}

# =========================================================
# TRAINING FUNCTION
# =========================================================
def train_ppo_wandb(
    env_name: str = "CarRacing-v3",
    wandb_project: str = "PPO-CarRacing",
    save_models: bool = True,
    models_dir: str = "models",
    eval_video_dir: str = "videos_eval",
    seed: int = 0,
    render: bool = False,
    use_wandb: bool = True
):
    """
    Train PPO agent on CarRacing-v3 with W&B logging.
    """
    # ---------------------------
    # Load configuration
    # ---------------------------
    if env_name not in PPO_CONFIG:
        raise ValueError(f"No PPO configuration found for {env_name}")
    
    cfg = PPO_CONFIG[env_name]
    
    # ---------------------------
    # Set seeds
    # ---------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # ---------------------------
    # Device setup
    # ---------------------------
    device = get_device()
    print(f"\n{'='*60}")
    print(f"PPO Training on {env_name}")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    # ---------------------------
    # W&B initialization
    # ---------------------------
    if use_wandb:
        wandb.login()
        run_name = f"PPO_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            config=cfg
        )
    
    # ---------------------------
    # Create directories
    # ---------------------------
    if save_models:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(eval_video_dir, exist_ok=True)
    
    # ---------------------------
    # Environment setup
    # ---------------------------
    env = CarRacingWrapper(
        env_name=env_name,
        img_stack=cfg["img_stack"],
        action_repeat=cfg["action_repeat"],
        seed=seed
    )
    
    # ---------------------------
    # Agent setup
    # ---------------------------
    agent = PPOAgent(
        img_stack=cfg["img_stack"],
        action_dim=3,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_param=cfg["clip_param"],
        ppo_epochs=cfg["ppo_epochs"],
        batch_size=cfg["batch_size"],
        buffer_capacity=cfg["buffer_capacity"],
        max_grad_norm=cfg["max_grad_norm"],
        value_loss_coef=cfg["value_loss_coef"],
        entropy_coef=cfg["entropy_coef"],
        device=device
    )
    
    # ---------------------------
    # Training loop (timestep-based)
    # ---------------------------
    running_score = 0
    best_score = -float('inf')
    episode_rewards = []
    
    total_timesteps_cfg = cfg["total_timesteps"]
    start_timesteps = cfg.get("start_timesteps", 0)
    max_ep_len = cfg["max_ep_len"]
    
    t = 0  # Total timesteps counter
    episode = 0
    last_save_t = 0
    last_eval_t = 0
    
    print("\nStarting training...")
    print(f"Total timesteps: {total_timesteps_cfg:,}")
    print(f"Random exploration for first {start_timesteps:,} timesteps")
    print("-" * 60)
    
    while t < total_timesteps_cfg:
        episode += 1
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for ep_t in range(max_ep_len):
            # Select action
            # PPO usually does NOT use pure random actions at the start because it's on-policy.
            # We trust the initialized policy (which is random anyway) to collect valid data.
            # However, if you prefer explicit random warmup, we keep your logic but reduce the duration.
            if t < start_timesteps:
                action = np.random.random(3).astype(np.float32)
                action_log_prob = 0.0
                value = 0.0
            else:
                action, action_log_prob, value = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, truncated, info = env.step(action)
            
            if render:
                env.render()
            
            # Store transition (Store ALL transitions, even during warmup, or PPO won't have data to start)
            agent.store_transition(
                state, action, action_log_prob, reward, value, 
                done or truncated
            )
            
            episode_reward += reward
            episode_steps += 1
            t += 1
            state = next_state
            
            # Update if buffer is full
            if len(agent.buffer) >= cfg["buffer_capacity"]:
                update_info = agent.update()
                if update_info and use_wandb:
                    wandb.log({
                        "policy_loss": update_info["policy_loss"],
                        "value_loss": update_info["value_loss"],
                        "entropy": update_info["entropy"],
                        "training_step": update_info["training_step"],
                        "timestep": t
                    })
            
            # Periodic save (timestep-based)
            if save_models and t - last_save_t >= cfg["save_interval"]:
                save_path = os.path.join(models_dir, f"ppo_{env_name}_t{t}.pth")
                agent.save(save_path)
                last_save_t = t
            
            # Periodic evaluation (timestep-based)
            if t - last_eval_t >= cfg["eval_interval"]:
                eval_score = evaluate_agent(
                    agent, 
                    env_name,
                    cfg,
                    num_episodes=cfg["eval_episodes"],
                    record_video=(save_models and (t // cfg["eval_interval"]) % 2 == 0),
                    video_dir=os.path.join(eval_video_dir, env_name),
                    episode_num=episode
                )
                print(f"  [t={t:,}] Evaluation Score: {eval_score:.2f}")
                if use_wandb:
                    wandb.log({"eval_score": eval_score, "timestep": t})
                last_eval_t = t
            
            if done or truncated or info.get('die', False):
                break
            
            if t >= total_timesteps_cfg:
                break
        
        # Track metrics
        episode_rewards.append(episode_reward)
        running_score = 0.99 * running_score + 0.01 * episode_reward
        
        # Logging
        if use_wandb:
            wandb.log({
                "episode": episode,
                "episode_reward": episode_reward,
                "running_score": running_score,
                "episode_steps": episode_steps,
                "timestep": t
            })
        
        if episode % cfg["log_interval"] == 0:
            avg_reward = np.mean(episode_rewards[-cfg["log_interval"]:])
            print(f"Episode {episode:5d} | "
                  f"Timestep {t:8,} | "
                  f"Score: {episode_reward:8.2f} | "
                  f"Avg(10): {avg_reward:8.2f} | "
                  f"Running: {running_score:8.2f}")
        
        # Save best model
        if running_score > best_score and t > start_timesteps:
            best_score = running_score
            if save_models:
                save_path = os.path.join(models_dir, f"ppo_{env_name}_best.pth")
                agent.save(save_path)
        
        # Check if solved
        if running_score > 900:
            print(f"\n{'='*60}")
            print(f"SOLVED! Running score: {running_score:.2f} at timestep {t:,}")
            print(f"{'='*60}")
            if save_models:
                save_path = os.path.join(models_dir, f"ppo_{env_name}_solved.pth")
                agent.save(save_path)
            break
    
    env.close()
    
    if use_wandb:
        wandb.finish()
    
    print("\nTraining completed!")
    return agent


def evaluate_agent(
    agent: PPOAgent,
    env_name: str,
    cfg: dict,
    num_episodes: int = 5,
    record_video: bool = False,
    video_dir: str = None,
    episode_num: int = 0
) -> float:
    """
    Evaluate PPO agent using the EXACT SAME wrapper as training.
    """
    # 1. Create base environment
    if record_video and video_dir:
        os.makedirs(video_dir, exist_ok=True)
        # We must use the same wrapper, so we wrap the base env first
        env = CarRacingWrapper(
            env_name=env_name,
            img_stack=cfg["img_stack"],
            action_repeat=cfg["action_repeat"],
            seed=0
        )
        # Inject RecordVideo wrapper properly (Tricky part: Wrapper -> RecordVideo -> Env)
        # Easier method: Just use the wrapper normally, but capture frames manually if needed.
        # OR: Just rely on the wrapper's render() if it supports it.
        
        # Simpler approach for now: Use the standard wrapper logic
        # Ideally, we pass the 'render_mode' to the wrapper if supported.
    
    # 2. Use the SAME wrapper class (Critical Fix)
    eval_env = CarRacingWrapper(
        env_name=env_name,
        img_stack=cfg["img_stack"],
        action_repeat=cfg["action_repeat"],
        seed=42 + episode_num # Different seed to test generalization
    )

    total_rewards = []

    for _ in range(num_episodes):
        state = eval_env.reset()
        episode_reward = 0
        
        for _ in range(cfg["max_ep_len"]):
            # Deterministic=True is standard for evaluation (removes randomness)
            action, _, _ = agent.select_action(state, deterministic=True)
            
            # The wrapper handles the action transformation automatically
            state, reward, done, truncated, info = eval_env.step(action)
            episode_reward += reward
            
            if done or truncated or info.get('die', False):
                break
        
        total_rewards.append(episode_reward)

    eval_env.close()
    return np.mean(total_rewards)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on CarRacing-v3")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--models-dir", type=str, default="models", help="Models save directory")
    parser.add_argument("--videos-dir", type=str, default="videos_eval", help="Videos save directory")
    
    args = parser.parse_args()
    
    agent = train_ppo_wandb(
        env_name=args.env,
        save_models=True,
        models_dir=args.models_dir,
        eval_video_dir=args.videos_dir,
        seed=args.seed,
        render=args.render,
        use_wandb=not args.no_wandb
    )