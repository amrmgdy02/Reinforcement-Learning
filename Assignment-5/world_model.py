import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import OneHotCategorical
from torch.distributions.kl import kl_divergence
from torch.cuda.amp import autocast, GradScaler
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import random
import wandb
from PIL import Image
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def lambda_return(rewards, values, continues, bootstrap, lambda_=0.95):
    """Compute lambda returns with numerical stability."""
    next_values = torch.cat([values[:, 1:], bootstrap[:, None]], dim=1)
    targets = rewards + continues * next_values
    
    outputs = []
    last = bootstrap
    for t in reversed(range(rewards.shape[1])):
        last = targets[:, t] + continues[:, t] * lambda_ * (last - next_values[:, t])
        outputs.append(last)
    
    returns = torch.stack(list(reversed(outputs)), dim=1)
    return returns


class Encoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64 → 32
            nn.LayerNorm([32, 32, 32]), 
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 → 16
            nn.LayerNorm([64, 16, 16]),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 → 8
            nn.LayerNorm([128, 8, 8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8 → 4
            nn.LayerNorm([256, 4, 4]),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(256 * 4 * 4, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, obs):
        # Normalize once with clamping for stability
        x = obs.float().clamp(0, 255) * (1.0 / 255.0)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = self.ln(x)
        return x

class Decoder(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 256 * 4 * 4),
            nn.LayerNorm(256 * 4 * 4)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4 → 8
            nn.LayerNorm([128, 8, 8]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 → 16
            nn.LayerNorm([64, 16, 16]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 → 32
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 32 → 64
        )
        
    def forward(self, feat):
        x = self.fc(feat)
        x = x.view(-1, 256, 4, 4)
        x = self.deconv(x)
        return torch.sigmoid(x)  # Bounded output for stability

class RSSM(nn.Module):
    def __init__(self, action_dim, embed_dim, deter_size=1024, stoch_size=32, 
                 hidden_size=512, num_classes=32):
        super().__init__()
        self.deter_size = deter_size
        self.stoch_size = stoch_size
        self.num_classes = num_classes
        
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, stoch_size * num_classes) # outputs flat logits
        )
        
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + embed_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, stoch_size * num_classes) # outputs flat logits
        )
        
        self.input_ln = nn.LayerNorm(stoch_size * num_classes + action_dim)
        # Input: previous stochastic state + action
        # Output: new deterministic state
        self.gru = nn.GRUCell(stoch_size * num_classes + action_dim, deter_size) # ht​ = GRU([zt−1​,at−1​], ht−1​)
        
    def initial_state(self, batch_size):
        return {
            'deter': torch.zeros(batch_size, self.deter_size, device=device),
            'stoch': torch.zeros(batch_size, self.stoch_size, self.num_classes, device=device)
        }
    
    def get_dist(self, logits):
        logits = logits.reshape(*logits.shape[:-1], self.stoch_size, self.num_classes) # reshape flat logit into (B, stoch_size, num_classes)
        # Temperature scaling for more stable training
        logits = logits / 1.0
        return OneHotCategorical(logits=logits) # Discrete latents better capture mode switches (ball goes left vs right).
    
    def observe(self, embed, action, state): # Used during training with real observations.
        stoch_flat = state['stoch'].reshape(state['stoch'].shape[0], -1)
        gru_input = self.input_ln(torch.cat([stoch_flat, action], dim=-1))
        deter = self.gru(gru_input, state['deter'])
        
        post_logits = self.posterior_net(torch.cat([deter, embed], dim=-1))
        post_dist = self.get_dist(post_logits)
        stoch = post_dist.sample() + post_dist.probs - post_dist.probs.detach()
        
        prior_logits = self.prior_net(deter)
        
        return {'deter': deter, 'stoch': stoch}, prior_logits, post_logits
    
    def imagine(self, action, state): # Used for planning & policy learning.
        stoch_flat = state['stoch'].reshape(state['stoch'].shape[0], -1)
        gru_input = self.input_ln(torch.cat([stoch_flat, action], dim=-1))
        deter = self.gru(gru_input, state['deter'])
        
        prior_logits = self.prior_net(deter)
        prior_dist = self.get_dist(prior_logits)
        stoch = prior_dist.sample() + prior_dist.probs - prior_dist.probs.detach()
        
        return {'deter': deter, 'stoch': stoch}
    
    def get_feat(self, state):
        stoch = state['stoch']
        deter = state['deter']
        
        if stoch.dim() == 3:
            stoch_flat = stoch.reshape(stoch.shape[0], -1)
        elif stoch.dim() == 4:
            stoch_flat = stoch.reshape(*stoch.shape[:2], -1)
        else:
            raise ValueError(f"Unexpected stoch dimensions: {stoch.shape}")
        
        return torch.cat([deter, stoch_flat], dim=-1)

class Actor(nn.Module):
    def __init__(self, feat_size, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim)
        )
        
    def forward(self, feat):
        logits = self.net(feat)
        return OneHotCategorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, feat_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        
    def forward(self, feat):
        return self.net(feat).squeeze(-1)

# DreamerV3 Agent
class DreamerV3Agent:
    def __init__(self, action_dim, obs_shape=(3, 64, 64), lr=1e-4):  # Lower LR
        self.action_dim = action_dim
        
        # Create encoder to get actual embed dimension
        self.encoder = Encoder().to(device)
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *obs_shape, device=device)
            embed_dim = self.encoder(dummy_obs).shape[-1]
        print(f"📐 Encoder output dimension: {embed_dim}")
        
        feat_size = 1024 + 32 * 32  # Updated: deter_size + stoch_size * num_classes
        
        self.rssm = RSSM(action_dim, embed_dim=embed_dim).to(device)
        self.decoder = Decoder(feat_size).to(device)
        self.reward_pred = Critic(feat_size).to(device)
        self.continue_pred = Critic(feat_size).to(device)
        self.actor = Actor(feat_size, action_dim).to(device)
        self.critic = Critic(feat_size).to(device)
        
        # Separate optimizers with different learning rates
        world_params = (list(self.encoder.parameters()) + 
                       list(self.rssm.parameters()) + 
                       list(self.decoder.parameters()) + 
                       list(self.reward_pred.parameters()) + 
                       list(self.continue_pred.parameters()))
        
        self.world_opt = optim.AdamW(world_params, lr=lr, weight_decay=1e-6, 
                                      eps=1e-8, betas=(0.9, 0.999))
        self.actor_opt = optim.AdamW(self.actor.parameters(), lr=lr * 0.8,  # Slower actor
                                      weight_decay=1e-6, eps=1e-8)
        self.critic_opt = optim.AdamW(self.critic.parameters(), lr=lr, 
                                       weight_decay=1e-6, eps=1e-8)
        
        # Mixed precision training
        self.scaler = GradScaler() if device.type == 'cuda' else None
        self.use_amp = device.type == 'cuda'
        
        # EMA for target critic (improves stability)
        self.target_critic = Critic(feat_size).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic_tau = 0.02
        
    def update_target_critic(self):
        """Soft update of target critic."""
        for param, target_param in zip(self.critic.parameters(), 
                                       self.target_critic.parameters()):
            target_param.data.copy_(
                self.target_critic_tau * param.data + 
                (1 - self.target_critic_tau) * target_param.data
            )
    
    def act(self, obs, state, training=True):
        with torch.no_grad():
            embed = self.encoder(torch.FloatTensor(obs).unsqueeze(0).to(device))
            
            if state is None:
                state = self.rssm.initial_state(1)
                action_dummy = torch.zeros(1, self.action_dim, device=device)
                state, _, _ = self.rssm.observe(embed, action_dummy, state)
            
            feat = self.rssm.get_feat(state)
            action_dist = self.actor(feat)
            
            if training:
                action = action_dist.sample()
            else:
                action = F.one_hot(action_dist.probs.argmax(dim=-1), 
                                  self.action_dim).float()
            
        return action.cpu().numpy()[0].argmax(), state
    
    def train_batch(self, obs, actions, rewards, dones):
        B, T = obs.shape[:2]
        
        # Training
        with autocast(enabled=self.use_amp):
            # Encode observations
            embed = self.encoder(obs.reshape(B * T, *obs.shape[2:]))
            embed = embed.reshape(B, T, -1)
            
            # World Model Rollout
            state = self.rssm.initial_state(B)
            states, priors, posts = [], [], []
            
            for t in range(T):
                state, prior, post = self.rssm.observe(embed[:, t], actions[:, t], state)
                states.append(state)
                priors.append(prior)
                posts.append(post)
            
            deter = torch.stack([s["deter"] for s in states], dim=1)
            stoch = torch.stack([s["stoch"] for s in states], dim=1)
            feat = self.rssm.get_feat({"deter": deter, "stoch": stoch})
            
            # World Model Losses
            obs_flat = obs.reshape(B * T, *obs.shape[2:]).float() / 255.0
            recon = self.decoder(feat.reshape(B * T, -1))
            recon_loss = F.mse_loss(recon, obs_flat)
            
            feat_flat = feat.reshape(B * T, -1)
            pred_reward = self.reward_pred(feat_flat).reshape(B, T)
            # Use symlog for reward prediction stability
            reward_loss = F.mse_loss(pred_reward, symlog(rewards))
            
            pred_continue = self.continue_pred(feat_flat).reshape(B, T)
            continue_target = 1.0 - dones.float()
            continue_loss = F.binary_cross_entropy_with_logits(pred_continue, continue_target)
            
            # KL balancing (DreamerV3 style)
            prior_dist = self.rssm.get_dist(torch.stack(priors, 1))
            post_dist = self.rssm.get_dist(torch.stack(posts, 1))
            
            kl_post_prior = kl_divergence(post_dist, prior_dist).mean()
            kl_prior_post = kl_divergence(prior_dist, post_dist).mean()
            
            # KL balancing: 0.8 * max(1.0, kl_post_prior) + 0.2 * kl_prior_post
            kl_loss = 0.8 * torch.maximum(kl_post_prior, torch.tensor(1.0, device=device)) + \
                     0.2 * kl_prior_post
            
            # Weighted losses
            world_loss = 1.0 * recon_loss + 1.0 * reward_loss + \
                        1.0 * continue_loss + 0.1 * kl_loss
        
        # World model update with gradient scaling
        self.world_opt.zero_grad()
        if self.use_amp:
            self.scaler.scale(world_loss).backward()
            self.scaler.unscale_(self.world_opt)
            world_params = (list(self.encoder.parameters()) + 
                           list(self.rssm.parameters()) + 
                           list(self.decoder.parameters()) + 
                           list(self.reward_pred.parameters()) + 
                           list(self.continue_pred.parameters()))
            torch.nn.utils.clip_grad_norm_(world_params, 100.0)
            self.scaler.step(self.world_opt)
            self.scaler.update()
        else:
            world_loss.backward()
            world_params = (list(self.encoder.parameters()) + 
                           list(self.rssm.parameters()) + 
                           list(self.decoder.parameters()) + 
                           list(self.reward_pred.parameters()) + 
                           list(self.continue_pred.parameters()))
            torch.nn.utils.clip_grad_norm_(world_params, 100.0)
            self.world_opt.step()
        
        # Imagination Rollout
        with torch.no_grad():
            # Start from last state
            imag_state = {
                "deter": deter[:, -1].detach().float(),
                "stoch": stoch[:, -1].detach().float()
            }
            
            imag_states = [imag_state]
            horizon = 15
            
            for _ in range(horizon):
                feat_i = self.rssm.get_feat(imag_state)
                action = self.actor(feat_i).sample()
                imag_state = self.rssm.imagine(action, imag_state)
                imag_states.append(imag_state)
            
            imag_deter = torch.stack([s["deter"] for s in imag_states], dim=1)
            imag_stoch = torch.stack([s["stoch"] for s in imag_states], dim=1)
            imag_feat = self.rssm.get_feat({"deter": imag_deter, "stoch": imag_stoch})
        
        # Actor Update
        imag_feat_flat = imag_feat.reshape(-1, imag_feat.shape[-1])
        imag_reward = self.reward_pred(imag_feat_flat).reshape(B, horizon + 1)
        imag_value = self.target_critic(imag_feat_flat).reshape(B, horizon + 1)  # Use target
        imag_continue = torch.sigmoid(self.continue_pred(imag_feat_flat)).reshape(B, horizon + 1)
        
        # Compute lambda returns
        returns = lambda_return(
            imag_reward[:, :-1],
            imag_value[:, :-1],
            imag_continue[:, :-1],
            imag_value[:, -1],
            lambda_=0.95
        )
        
        # Discount accumulation
        discount = torch.cumprod(
            torch.cat([torch.ones(B, 1, device=device), imag_continue[:, :-1]], dim=1),
            dim=1
        )
        
        # Actor loss with entropy bonus
        actor_dist = self.actor(imag_feat[:, :-1].reshape(-1, imag_feat.shape[-1]))
        entropy = actor_dist.entropy().reshape(B, horizon).mean()
        
        actor_loss = -(discount[:, :-1] * returns.detach()).mean() - 0.001 * entropy
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()
        
        # Critic Update
        imag_feat_detached = imag_feat.detach()
        imag_value_current = self.critic(
            imag_feat_detached.reshape(-1, imag_feat_detached.shape[-1])
        ).reshape(B, horizon + 1)
        
        critic_loss = F.mse_loss(imag_value_current[:, :-1], returns.detach())
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()
        
        # Update target critic
        self.update_target_critic()
        
        return {
            "world_loss": world_loss.item(),
            "recon_loss": recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "kl_loss": kl_loss.item(),
            "kl_post_prior": kl_post_prior.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

# Replay Buffer

class ReplayBuffer:
    def __init__(self, capacity=100000, seq_len=50, obs_shape=(3, 64, 64), action_dim=6):
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, obs, act, rew, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        if self.size < self.seq_len + 1:
            return None
        
        # Avoid sampling across episode boundaries
        valid_indices = []
        for _ in range(batch_size * 10):
            idx = np.random.randint(0, self.size - self.seq_len - 1)
            # Check if sequence doesn't cross episode boundary
            if not np.any(self.done_buf[idx:idx+self.seq_len-1]):
                valid_indices.append(idx)
                if len(valid_indices) == batch_size:
                    break
        
        if len(valid_indices) < batch_size:
            # Fallback to random sampling if not enough valid sequences
            valid_indices = np.random.randint(0, self.size - self.seq_len - 1, 
                                            size=batch_size)
        
        indices = np.array(valid_indices)
        
        obs_batch = np.stack([self.obs_buf[i:i+self.seq_len] for i in indices])
        act_batch = np.stack([self.act_buf[i:i+self.seq_len] for i in indices])
        rew_batch = np.stack([self.rew_buf[i:i+self.seq_len] for i in indices])
        done_batch = np.stack([self.done_buf[i:i+self.seq_len] for i in indices])
        
        return (
            torch.from_numpy(obs_batch).to(device),
            torch.from_numpy(act_batch).to(device),
            torch.from_numpy(rew_batch).to(device),
            torch.from_numpy(done_batch).to(device)
        )

def preprocess_obs(obs):
    """Optimized preprocessing."""
    img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
    return np.transpose(np.array(img), (2, 0, 1))

# Training Loop

def train(resume_from=None, max_episodes=500):
    wandb.init(
        project="dreamer-v3-spaceinvaders-optimized",
        config={
            "lr": 1e-4,
            "horizon": 15,
            "batch_size": 16,  # Larger batches for stability
            "seq_len": 50,
            "buffer_size": 100000,
            "train_every": 4,  # Train more frequently
            "mixed_precision": True,
            "target_critic_tau": 0.02,
        },
        resume="allow" if resume_from else False
    )
    
    env_name = 'SpaceInvadersNoFrameskip-v4'
    env = gym.make(env_name, render_mode="rgb_array")
    print(f"✅ Successfully created environment: {env_name}")
    
    try:
        env = RecordVideo(env, "./videos", episode_trigger=lambda x: x % 50 == 0)
    except Exception as e:
        print(f"⚠️ Video recording failed: {e}, continuing without recording...")
    
    obs_sample, _ = env.reset()
    obs_preprocessed = preprocess_obs(obs_sample)
    obs_shape = obs_preprocessed.shape
    print(f"📐 Preprocessed observation shape: {obs_shape}")
    
    agent = DreamerV3Agent(action_dim=env.action_space.n, obs_shape=obs_shape)
    buffer = ReplayBuffer(obs_shape=obs_shape, action_dim=env.action_space.n, 
                         capacity=100000)
    
    global_step = 0
    episode_rewards = deque(maxlen=100)
    best_reward = -float('inf')
    start_episode = 0
    
    if resume_from and os.path.exists(resume_from):
        print(f"📂 Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.target_critic.load_state_dict(checkpoint.get('target_critic', 
                                                           checkpoint['critic']))
        agent.encoder.load_state_dict(checkpoint['encoder'])
        agent.rssm.load_state_dict(checkpoint['rssm'])
        if 'decoder' in checkpoint:
            agent.decoder.load_state_dict(checkpoint['decoder'])
        if 'reward_pred' in checkpoint:
            agent.reward_pred.load_state_dict(checkpoint['reward_pred'])
        if 'continue_pred' in checkpoint:
            agent.continue_pred.load_state_dict(checkpoint['continue_pred'])
        
        start_episode = checkpoint.get('episode', 0) + 1
        best_reward = checkpoint.get('best_reward', -float('inf'))
        print(f"✅ Resuming from episode {start_episode}, best reward: {best_reward:.1f}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    import time
    start_time = time.time()
    
    for episode in range(start_episode, start_episode + max_episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        state = None
        ep_reward = 0
        done = False
        step = 0
        
        while not done:
            action_idx, state = agent.act(obs, state, training=True)
            act_onehot = np.zeros(env.action_space.n, dtype=np.float32)
            act_onehot[action_idx] = 1.0
            
            next_obs, reward, term, trunc, _ = env.step(action_idx)
            done = term or trunc
            next_obs = preprocess_obs(next_obs)
            
            buffer.add(obs, act_onehot, reward, done)
            
            obs = next_obs
            ep_reward += reward
            step += 1
            global_step += 1
            
            # Train more frequently with smaller batches for stability
            if step % 4 == 0 and buffer.size > 5000:
                batch = buffer.sample(16)
                if batch:
                    metrics = agent.train_batch(*batch)
                    
                    if global_step % 100 == 0:
                        wandb.log(metrics, step=global_step)
            
            # Log reconstruction less frequently
            if global_step % 1000 == 0:
                with torch.no_grad():
                    batch = buffer.sample(1)
                    if batch:
                        sample_obs = batch[0][0, 0]
                        sample_obs_tensor = sample_obs.unsqueeze(0).to(device)
                        
                        embed = agent.encoder(sample_obs_tensor)
                        state_sample = agent.rssm.initial_state(1)
                        action_dummy = torch.zeros(1, agent.action_dim, device=device)
                        state_sample, _, _ = agent.rssm.observe(embed, action_dummy, state_sample)
                        feat = agent.rssm.get_feat(state_sample)
                        recon = agent.decoder(feat)
                        
                        original_img = (sample_obs.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        recon_img = (recon[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        wandb.log({
                            "reconstruction/original": wandb.Image(original_img),
                            "reconstruction/reconstructed": wandb.Image(recon_img)
                        }, step=global_step)
        episode_rewards.append(ep_reward)
        avg_reward = np.mean(episode_rewards)
        wandb.log({"episode_reward": ep_reward, "avg_reward_100": avg_reward}, step=global_step)
        print(f"Episode {episode} | Reward: {ep_reward:.1f} | Avg Reward (100): {avg_reward:.1f} | Steps: {step} | Total Steps: {global_step}")
        
        # Save best model
        if avg_reward > best_reward:
            best_reward = avg_reward
            checkpoint = {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'target_critic': agent.target_critic.state_dict(),
                'encoder': agent.encoder.state_dict(),
                'rssm': agent.rssm.state_dict(),
                'decoder': agent.decoder.state_dict(),
                'reward_pred': agent.reward_pred.state_dict(),
                'continue_pred': agent.continue_pred.state_dict(),
                'episode': episode,
                'best_reward': best_reward
            }
            torch.save(checkpoint, f"checkpoints/dreamer_v3_best.pth")
            print(f"💾 Saved new best model with avg reward: {best_reward:.1f}")
        # Periodic checkpointing
        if episode % 50 == 0:
            checkpoint = {
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'target_critic': agent.target_critic.state_dict(),
                'encoder': agent.encoder.state_dict(),
                'rssm': agent.rssm.state_dict(),
                'decoder': agent.decoder.state_dict(),
                'reward_pred': agent.reward_pred.state_dict(),
                'continue_pred': agent.continue_pred.state_dict(),
                'episode': episode,
                'best_reward': best_reward
            }
            torch.save(checkpoint, f"checkpoints/dreamer_v3_ep{episode}.pth")
            print(f"💾 Saved checkpoint at episode {episode}")
            
    env.close()
if __name__ == "__main__":
    train()