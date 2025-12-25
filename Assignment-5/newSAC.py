# train_sac_spaceinvaders_full_fixed.py
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import ale_py  # <<---- Add this import
import wandb
from torch.distributions.categorical import Categorical

# -------------------------
# Helper functions
# -------------------------
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_space, device):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.device = device
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.ptr] = obs
        self.next_observations[self.ptr] = next_obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.observations[idxs], dtype=torch.float32, device=self.device) / 255.0,
            torch.tensor(self.next_observations[idxs], dtype=torch.float32, device=self.device) / 255.0,
            torch.tensor(self.actions[idxs], dtype=torch.int64, device=self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device),
        )

# -------------------------
# CNN Actor & Critic
# -------------------------
class Actor(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, 8, 4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, 2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, 1)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.fc = layer_init(nn.Linear(n_flatten, 512))
        self.logits = layer_init(nn.Linear(512, n_actions))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc(x))
        return self.logits(x)

    def get_action(self, obs, deterministic=False):
        logits = self(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=1)
        else:
            action = dist.sample()
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, dist.probs

class SoftQNetwork(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, 8, 4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, 2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, 1)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.fc = layer_init(nn.Linear(n_flatten, 512))
        self.q = layer_init(nn.Linear(512, n_actions))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc(x))
        return self.q(x)

# -------------------------
# Save / Load
# -------------------------
# -------------------------
# Save / Load
# -------------------------
def save_checkpoint(actor, q1, q2, log_alpha, step, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),  # save as tensor
        "step": step
    }, os.path.join(save_dir, f"checkpoint_{step}.pt"))
    print(f"[Checkpoint] Saved at step {step}")

def load_checkpoint(actor, q1, q2, log_alpha, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])
    # Restore log_alpha ensuring its dtype matches the current tensor (avoid Double/Float mismatch)
    log_alpha.data.copy_(checkpoint["log_alpha"].to(device).to(log_alpha.dtype))  # restore tensor value
    print(f"[Checkpoint] Loaded from step {checkpoint['step']}")


# -------------------------
# Environment Wrapper
# -------------------------
from atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv

# Try to import cv2 for ResizeObservation; if not available, provide a small PIL-based fallback
try:
    import cv2  # noqa: F401
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    from PIL import Image

    class ResizeObservationPIL(gym.ObservationWrapper):
        def __init__(self, env, shape):
            super().__init__(env)
            self.shape = shape
            old_space = env.observation_space
            channels = old_space.shape[2] if len(old_space.shape) == 3 else 1
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(shape[0], shape[1], channels),
                dtype=old_space.dtype
            )

        def observation(self, observation):
            img = Image.fromarray(observation)
            img = img.resize(self.shape, resample=Image.BILINEAR)
            return np.array(img)


def make_env(env_id, seed, capture_video=False, video_dir="videos"):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array" if capture_video else None)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, video_dir)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        if CV2_AVAILABLE:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
        else:
            env = ResizeObservationPIL(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk

# -------------------------
# Training Function
# -------------------------
def train(env_id="SpaceInvadersNoFrameskip-v4",
          total_timesteps=5_000_000,
          buffer_size=200_000,
          batch_size=64,
          learning_starts=25_000,
          update_freq=4,
          gamma=0.99,
          tau=0.005,
          alpha=0.1,
          lr=3e-4,
          seed=0,
          save_dir="models/sac_spaceinvaders",
          video_interval=50_000,
          use_wandb=True,
          wandb_project="Atari-SAC-Bonus"):

    device = get_device()
    env = gym.vector.SyncVectorEnv([make_env(env_id, seed)])
    obs_shape = env.single_observation_space.shape
    n_actions = env.single_action_space.n

    # ---------------------------
    # W&B initialization (follow trainSAC.py pattern)
    # ---------------------------
    if use_wandb:
        try:
            wandb.login()
            cfg = {
                "env": env_id,
                "total_timesteps": total_timesteps,
                "batch_size": batch_size,
                "buffer_size": buffer_size,
                "lr": lr,
                "gamma": gamma,
                "tau": tau,
                "alpha": alpha,
                "seed": seed,
            }
            wandb.init(project=wandb_project, name=f"SAC_{env_id}", config=cfg)
        except Exception as e:
            print(f"wandb init failed: {e}")

    actor = Actor(obs_shape, n_actions).to(device)
    q1 = SoftQNetwork(obs_shape, n_actions).to(device)
    q2 = SoftQNetwork(obs_shape, n_actions).to(device)
    q1_target = SoftQNetwork(obs_shape, n_actions).to(device)
    q2_target = SoftQNetwork(obs_shape, n_actions).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    q_optimizer = optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    # Make sure log_alpha is float32 to avoid dtype promotion to Double during computations
    log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=device)
    a_optimizer = optim.Adam([log_alpha], lr=lr)

    replay_buffer = ReplayBuffer(buffer_size, obs_shape, env.single_action_space, device)

    obs, _ = env.reset(seed=seed)
    start_time = time.time()
    episode_reward = 0.0

    # Initialize loss placeholders so logging (and wandb) before the first update doesn't crash
    q_loss = torch.tensor(0.0, device=device)
    actor_loss = torch.tensor(0.0, device=device)

    for step in range(1, total_timesteps + 1):
        # -------------------
        # Select action
        # -------------------
        if step < learning_starts:
            action = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
            # Only permute if channels are in the last dimension (NHWC). Some wrappers already return NCHW.
            if obs_tensor.ndim == 4 and obs_tensor.shape[-1] in (1, 3, 4, 12):  # NHWC -> NCHW
                obs_tensor = obs_tensor.permute(0, 3, 1, 2)
            with torch.no_grad():
                action, _, _ = actor.get_action(obs_tensor)
                action = action.cpu().numpy()

        # -------------------
        # Step environment
        # -------------------
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)
        episode_reward += reward[0]

        # If the environment provides episode stats in the info dict, log them per-episode
        if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict) and 'episode' in info[0]:
            ep_info = info[0]['episode']
            ep_ret = ep_info.get('r')
            ep_len = ep_info.get('l')
            print(f"Episode finished at step {step}: return={ep_ret}, length={ep_len}")
            if use_wandb:
                try:
                    wandb.log({"episode_reward": ep_ret, "episode_length": ep_len, "timestep": step}, step=step)
                except Exception:
                    pass

        replay_buffer.add(obs[0], next_obs[0], action[0], reward[0], done[0])
        obs = next_obs

        # -------------------
        # Update networks
        # -------------------
        # -------------------
        # Update networks
        # -------------------
        if step >= learning_starts and step % update_freq == 0 and replay_buffer.size >= batch_size:
            # NOTE: For SAC, it's often better to do multiple updates if update_freq > 1
            # But let's stick to your structure for now.
            states, next_states, actions, rewards, dones = replay_buffer.sample(batch_size)

            # --- CRITICAL FIX 1: Permute dimensions for PyTorch CNN (NHWC -> NCHW) ---
            # Gym/Atari returns (Batch, H, W, C), PyTorch needs (Batch, C, H, W)
            # Only permute if channels are in the last dimension (NHWC). Some rollouts already return NCHW.
            if states.ndim == 4 and states.shape[-1] in (1, 3, 4, 12):
                states = states.permute(0, 3, 1, 2)
            if next_states.ndim == 4 and next_states.shape[-1] in (1, 3, 4, 12):
                next_states = next_states.permute(0, 3, 1, 2)

            # Q target
            with torch.no_grad():
                # get_action returns: action_indices, log_probs (Batch, Acts), probs (Batch, Acts)
                _, next_log_pi, next_probs = actor.get_action(next_states)
                
                q1_next = q1_target(next_states)
                q2_next = q2_target(next_states)
                
                # Discrete SAC formulation: sum( probs * (minQ - alpha * log_pi) )
                min_q_next = torch.min(q1_next, q2_next)
                target_v = next_probs * (min_q_next - log_alpha.exp() * next_log_pi)
                target_v = target_v.sum(dim=1) # Sum over actions to get Value of state
                
                q_target = rewards + gamma * (1 - dones) * target_v

            # Q loss
            # Gather Q-values for the specific actions taken in the replay buffer
            q1_pred = q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            q2_pred = q2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Actor loss
            _, log_pi_new, probs_new = actor.get_action(states)
            with torch.no_grad():
                q1_val = q1(states)
                q2_val = q2(states)
                min_q_val = torch.min(q1_val, q2_val)
            
            # Discrete SAC Actor Loss: sum( probs * (alpha * log_pi - minQ) )
            actor_loss = (probs_new * (log_alpha.exp() * log_pi_new - min_q_val)).sum(dim=1).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # --- CRITICAL FIX 2: Correct Alpha Loss for Discrete ---
            # We want to adjust alpha based on the entropy of the current policy
            # Target entropy is usually -log(1/|A|)
            target_entropy = -np.log(1.0 / n_actions) * 0.3 # slight heuristic buffer
            
            # Alpha loss = -alpha * (entropy - target_entropy)
            # Entropy = - sum(probs * log_probs)
            # So inside expectation: sum(probs * (-alpha * (log_pi + target_entropy)))
            alpha_loss = (probs_new.detach() * (-log_alpha * (log_pi_new.detach() + target_entropy))).sum(dim=1).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()

            # Soft update targets
            for param, target_param in zip(q1.parameters(), q1_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(q2.parameters(), q2_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # -------------------
        # Logging
        # -------------------
        if step % 1000 == 0:
            elapsed = time.time() - start_time
            sps = step / elapsed
            print(f"[Train] Step {step}, SPS: {sps:.2f}, Q_loss: {q_loss.item():.3f}, "
                  f"Actor_loss: {actor_loss.item():.3f}, Alpha: {log_alpha.exp().item():.3f}, "
                  f"Reward: {episode_reward:.2f}")
            # Log to W&B if enabled
            if use_wandb:
                try:
                    wandb.log({
                        "step": step,
                        "q_loss": q_loss.item(),
                        "actor_loss": actor_loss.item(),
                        "alpha": log_alpha.exp().item(),
                        "reward": float(episode_reward),
                        "sps": sps,
                        "buffer_size": replay_buffer.size,
                    }, step=step)
                except Exception:
                    pass
            episode_reward = 0.0

        # -------------------
        # Save checkpoint + video
        # -------------------
        if step % video_interval == 0:
            save_checkpoint(actor, q1, q2, log_alpha, step, save_dir)
            print(f"[Video] Recording progress at step {step}...")

            # Evaluate and record video for one episode
            env_vid = gym.vector.SyncVectorEnv([make_env(env_id, seed, capture_video=True, video_dir=os.path.join(save_dir, "videos"))])
            obs_vid, _ = env_vid.reset(seed=seed)
            done_vid = False
            while not done_vid:
                obs_tensor_vid = torch.tensor(obs_vid, dtype=torch.float32, device=device) / 255.0
                # Only permute if channels are in the last dimension (NHWC). Some wrappers already return NCHW.
                if obs_tensor_vid.ndim == 4 and obs_tensor_vid.shape[-1] in (1, 3, 4, 12):
                    obs_tensor_vid = obs_tensor_vid.permute(0, 3, 1, 2)
                action_vid, _, _ = actor.get_action(obs_tensor_vid, deterministic=True)
                obs_vid, reward_vid, terminated_vid, truncated_vid, info_vid = env_vid.step(action_vid.cpu().numpy())
                done_vid = np.logical_or(terminated_vid, truncated_vid)[0]
            env_vid.close()

    env.close()
    print("Training done.")
    if use_wandb:
        try:
            wandb.finish()
        except Exception:
            pass


# -------------------------
# Evaluation Function
# -------------------------
@torch.no_grad()
def evaluate(actor, env_id="SpaceInvadersNoFrameskip-v4", device=None, episodes=100, seed=0, video_dir="videos/sac_eval"):
    device = device or get_device()
    print(f"Starting evaluation on {device} for {episodes} episodes...")
    env = gym.vector.SyncVectorEnv([make_env(env_id, seed, capture_video=True, video_dir=video_dir)])
    obs, _ = env.reset(seed=seed)
    returns = []

    for ep in range(episodes):
        done = False
        ep_return = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
            if obs_t.dim() == 4 and obs_t.shape[-1] in (1, 3, 4, 12):
                obs_t = obs_t.permute(0, 3, 1, 2)
            action, _, _ = actor.get_action(obs_t, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            ep_return += reward[0]
        returns.append(ep_return)
        print(f"[Eval] Episode {ep+1} return: {ep_return}")
        obs, _ = env.reset()
    env.close()
    avg_return = np.mean(returns)
    print(f"[Eval] Average Return over {episodes} episodes: {avg_return}")
    return avg_return

@torch.no_grad()
def evaluate_sac_agent(actor, env_id="SpaceInvadersNoFrameskip-v4", device=None,
                       num_episodes=10, base_seed=0, video_dir="videos/sac_eval_full",
                       use_wandb=True, wandb_project="Atari-SAC-Eval", wandb_name=None):
    """
    Evaluate SAC actor on full games (all lives), no reward clipping.
    
    Args:
        actor: Trained SAC actor network
        env_id: Environment ID
        device: PyTorch device
        num_episodes: Number of episodes to evaluate
        base_seed: Base seed for environment
        video_dir: Directory to save evaluation videos
        use_wandb: Whether to use Weights & Biases for logging
        wandb_project: W&B project name
        wandb_name: W&B run name (if None, auto-generate)
    """
    import os
    import glob
    import torch
    import numpy as np
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    os.makedirs(video_dir, exist_ok=True)
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize WandB for evaluation
    if use_wandb:
        try:
            wandb.login()
            # Create run name if not provided
            if wandb_name is None:
                wandb_name = f"SAC_Eval_{env_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Configuration for evaluation
            eval_config = {
                "env_id": env_id,
                "num_episodes": num_episodes,
                "base_seed": base_seed,
                "video_dir": video_dir,
                "device": str(device),
            }
            
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=eval_config,
                tags=["evaluation", "SAC", "Atari"],
                reinit=True  # Allow reinitialization if called multiple times
            )
            print(f"[WandB] Evaluation run initialized: {wandb_name}")
        except Exception as e:
            print(f"[WandB] Initialization failed: {e}")
            use_wandb = False
    
    episode_rewards = []
    episode_lengths = []
    num_videos = min(5, num_episodes)  # only save videos for the first 5 episodes
    
    for ep in range(num_episodes):
        # randomize seed per episode for different starts
        seed = base_seed + ep
        
        # Build environment
        env = gym.make(env_id, render_mode="rgb_array")
        
        capture_video = (ep < num_videos)
        video_subdir = os.path.join(video_dir, f"episode_{ep+1}") if capture_video else None
        if capture_video:
            env = gym.wrappers.RecordVideo(
                env, 
                video_subdir,
                episode_trigger=lambda x: True,  # Record every episode
                disable_logger=True
            )
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if CV2_AVAILABLE:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
        else:
            env = ResizeObservationPIL(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.action_space.seed(seed)
        obs, _ = env.reset(seed=seed)
        
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            if obs_tensor.dim() == 4 and obs_tensor.shape[-1] in (1,3,4,12):
                obs_tensor = obs_tensor.permute(0,3,1,2)
            action, _, _ = actor.get_action(obs_tensor, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        # Close env (ensures video flush)
        env.close()
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Length = {steps}")
        
        # Log episode results to WandB
        if use_wandb:
            try:
                wandb.log({
                    "episode_reward": total_reward,
                    "episode_length": steps,
                    "episode": ep + 1
                })
            except Exception as e:
                print(f"[WandB] Logging failed for episode {ep+1}: {e}")
        
        # Handle video upload to WandB
        if capture_video:
            try:
                mp4s = glob.glob(os.path.join(video_subdir, "**", "*.mp4"), recursive=True)
                if mp4s:
                    latest_mp4 = max(mp4s, key=os.path.getmtime)
                    
                    if use_wandb:
                        try:
                            wandb.log({
                                f"eval/video_ep_{ep+1}": wandb.Video(
                                    latest_mp4, 
                                    caption=f"Eval episode {ep+1} (Reward: {total_reward:.1f})", 
                                    fps=30, 
                                    format="mp4"
                                )
                            })
                        except Exception as e:
                            print(f"[WandB] Video upload failed for episode {ep+1}: {e}")
                    else:
                        print(f"Saved video to {latest_mp4}")
                else:
                    print(f"No video file found in {video_subdir} for episode {ep+1}")
            except Exception as e:
                print(f"Error while handling video for episode {ep+1}: {e}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results ({num_episodes} episodes):")
    print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min/Max Reward: {min_reward:.2f} / {max_reward:.2f}")
    print(f"  Average Length: {avg_length:.0f} steps")
    print(f"{'='*60}")
    
    # Create and save reward plot (Rewards per Episode only)
    plt.figure(figsize=(10, 6))
    episodes = np.arange(1, num_episodes + 1)
    plt.plot(episodes, episode_rewards, marker='o', color='tab:blue', linewidth=2)
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Mean: {avg_reward:.1f}')
    plt.fill_between(episodes, avg_reward - std_reward, avg_reward + std_reward, alpha=0.1, color='gray')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Reward per Episode — {env_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(video_dir, "rewards_per_episode.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved reward-per-episode plot to {plot_path}")

    # Log to WandB    
    # Log to WandB
    if use_wandb:
        try:
            # Log summary statistics
            wandb.log({
                "eval/avg_reward": avg_reward,
                "eval/std_reward": std_reward,
                "eval/min_reward": min_reward,
                "eval/max_reward": max_reward,
                "eval/avg_length": avg_length,
                "eval/total_reward": np.sum(episode_rewards),
            })
            
            # Log the plot as an image
            wandb.log({"eval/summary_plot": wandb.Image(plot_path)})
            
            # Create a table of episode results
            results_table = wandb.Table(columns=["Episode", "Reward", "Length"])
            for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
                results_table.add_data(i+1, reward, length)
            
            wandb.log({"eval/episode_results": results_table})
            
            # Finish the WandB run
            wandb.finish()
            print("[WandB] Evaluation run completed and logged")
            
        except Exception as e:
            print(f"[WandB] Final logging failed: {e}")
            if wandb.run is not None:
                wandb.finish()
    
    plt.close()
    
    return avg_reward, episode_rewards, episode_lengths

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    checkpoint_path = "models/sac_spaceinvaders/checkpoint_5000000.pt"
    env_id = "SpaceInvadersNoFrameskip-v4"
    num_episodes = 100
    base_seed = 0
    
    device = get_device()
    
    # Dummy env to get obs_shape and n_actions
    env_tmp = gym.vector.SyncVectorEnv([make_env(env_id, base_seed)])
    obs_shape = env_tmp.single_observation_space.shape
    n_actions = env_tmp.single_action_space.n
    env_tmp.close()
    
    # Load Actor
    actor = Actor(obs_shape, n_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    print(f"Loaded checkpoint from {checkpoint_path}")
    
    # Evaluate with enhanced logging
    avg_reward, all_rewards, all_lengths = evaluate_sac_agent(
        actor, 
        env_id=env_id, 
        device=device, 
        num_episodes=num_episodes, 
        base_seed=base_seed,
        use_wandb=True,  # Set to True to enable WandB logging
        wandb_project="Atari-SAC-Evaluation",
        wandb_name=f"SAC_{env_id}_Final_Eval"  # Custom name for the run
    )
    
    # Print final summary
    print(f"\n✅ Evaluation Complete!")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   All Rewards: {[f'{r:.1f}' for r in all_rewards]}")
