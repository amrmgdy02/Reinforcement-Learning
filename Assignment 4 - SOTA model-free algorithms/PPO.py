"""
PPO (Proximal Policy Optimization) with Clipping for CarRacing-v3

This implementation uses:
- CNN feature extractor for image observations
- Beta distribution for bounded continuous actions
- PPO clipping objective
- GAE (Generalized Advantage Estimation)
- Frame stacking for temporal information
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Tuple, List
import gymnasium as gym


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Rollout Buffer for PPO
# -------------------------
class RolloutBuffer:
    """Buffer to store rollout data for PPO updates"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def push(self, state, action, action_log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.action_log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.dones[:]
        
    def __len__(self):
        return len(self.states)


# -------------------------
# CNN Feature Extractor
# -------------------------
class CNNFeatureExtractor(nn.Module):
    """
    CNN for extracting features from stacked grayscale frames.
    Input: (batch, img_stack, 96, 96)
    Output: (batch, 256)
    """
    def __init__(self, img_stack=4):
        super(CNNFeatureExtractor, self).__init__()
        
        self.cnn_base = nn.Sequential(
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )  # Output shape: (256, 1, 1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
                    
    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, 256)
        return x


# -------------------------
# Actor-Critic Network
# -------------------------
class ActorCriticCNN(nn.Module):
    """
    Actor-Critic network with shared CNN backbone.
    Uses Beta distribution for bounded continuous actions.
    
    Actions for CarRacing:
    - steering: [-1, 1]
    - gas: [0, 1]  
    - brake: [0, 1]
    """
    def __init__(self, img_stack=4, action_dim=3):
        super(ActorCriticCNN, self).__init__()
        
        self.feature_extractor = CNNFeatureExtractor(img_stack)
        
        # Value head (Critic)
        self.value_head = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
        # Policy head (Actor) - outputs alpha and beta for Beta distribution
        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU()
        )
        
        # Alpha and Beta heads for Beta distribution (must be > 0)
        self.alpha_head = nn.Sequential(
            nn.Linear(100, action_dim),
            nn.Softplus()  # Ensures positive output
        )
        self.beta_head = nn.Sequential(
            nn.Linear(100, action_dim),
            nn.Softplus()  # Ensures positive output
        )
        
        self._initialize_linear_weights()
        
    def _initialize_linear_weights(self):
        for m in [self.value_head, self.fc, self.alpha_head, self.beta_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, x):
        """
        Returns:
            (alpha, beta): Parameters for Beta distribution
            value: State value estimate
        """
        features = self.feature_extractor(x)
        
        # Value
        value = self.value_head(features)
        
        # Policy (Beta distribution parameters)
        policy_features = self.fc(features)
        alpha = self.alpha_head(policy_features) + 1  # Alpha > 1 for more stable distribution
        beta = self.beta_head(policy_features) + 1    # Beta > 1 for more stable distribution
        
        return (alpha, beta), value


# -------------------------
# PPO Agent
# -------------------------
class PPOAgent:
    """
    PPO Agent with clipping objective for continuous control.
    Designed for image-based environments like CarRacing-v3.
    """
    def __init__(
        self,
        img_stack: int = 4,
        action_dim: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.1,
        ppo_epochs: int = 10,
        batch_size: int = 128,
        buffer_capacity: int = 2000,
        max_grad_norm: float = 0.5,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device=None
    ):
        """
        Args:
            img_stack: Number of stacked frames
            action_dim: Action space dimension (3 for CarRacing)
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter (epsilon)
            ppo_epochs: Number of PPO update epochs per rollout
            batch_size: Mini-batch size for updates
            buffer_capacity: Rollout buffer capacity
            max_grad_norm: Maximum gradient norm for clipping
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
        """
        self.device = device or get_device()
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.max_grad_norm = max_grad_norm
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim
        
        # Initialize network
        self.network = ActorCriticCNN(img_stack, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Training counter
        self.training_step = 0
        
        print(f"PPO Agent initialized on {self.device}")
        print(f"  - Image stack: {img_stack}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Clip param: {clip_param}")
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action using current policy.
        
        Args:
            state: Stacked grayscale frames (img_stack, 96, 96)
            deterministic: If True, use mean action
            
        Returns:
            action: Action in [0, 1] range (needs transformation for env)
            action_log_prob: Log probability of action
            value: State value estimate
        """
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            (alpha, beta), value = self.network(state_tensor)
            
            if deterministic:
                # Use mode of Beta distribution: (alpha - 1) / (alpha + beta - 2)
                # For alpha, beta > 1, mode exists
                action = (alpha - 1) / (alpha + beta - 2 + 1e-8)
                action = action.clamp(0, 1)
                action_log_prob = 0.0
            else:
                dist = Beta(alpha, beta)
                action = dist.sample()
                action_log_prob = dist.log_prob(action).sum(dim=1).item()
                
            action = action.squeeze(0).cpu().numpy()
            value = value.squeeze().item()
            
        return action, action_log_prob, value
    
    def store_transition(self, state, action, action_log_prob, reward, value, done):
        """Store transition in rollout buffer"""
        self.buffer.push(state, action, action_log_prob, reward, value, done)
        
    def compute_gae(self, next_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate for the state after the last stored state
            
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        
        gae = 0
        returns = []
        advantages = []
        
        # Compute GAE in reverse order
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_val = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_val * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def update(self) -> dict:
        """
        Perform PPO update using collected rollout data.
        
        Returns:
            Dictionary with loss information
        """
        if len(self.buffer) < self.batch_size:
            return {}
            
        self.training_step += 1
        
        # Get last state value for GAE computation
        with torch.no_grad():
            last_state = torch.from_numpy(self.buffer.states[-1]).float().to(self.device).unsqueeze(0)
            _, last_value = self.network(last_state)
            last_value = last_value.squeeze().item()
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(last_value)
        
        # Convert buffer data to tensors
        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(self.buffer.action_log_probs, dtype=torch.float32).to(self.device)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Mini-batch sampling
            sampler = BatchSampler(
                SubsetRandomSampler(range(len(self.buffer))),
                self.batch_size,
                drop_last=False
            )
            
            for indices in sampler:
                # Get mini-batch
                mb_states = states[indices]
                mb_actions = actions[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_returns = returns[indices]
                mb_advantages = advantages[indices]
                
                # Forward pass
                (alpha, beta), values = self.network(mb_states)
                values = values.squeeze()
                
                # Compute new log probs and entropy
                dist = Beta(alpha, beta)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.smooth_l1_loss(values, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Clear buffer after update
        self.buffer.clear()
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'training_step': self.training_step
        }
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        print(f"Model saved to {path}")
        
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {path}")


# -------------------------
# Environment Wrapper
# -------------------------
class CarRacingWrapper:
    """
    Environment wrapper for CarRacing-v3.
    
    Features:
    - Grayscale conversion and normalization
    - Frame stacking for temporal information
    - Action repeat for stability
    - Early termination if no progress
    """
    def __init__(
        self,
        env_name: str = "CarRacing-v3",
        img_stack: int = 4,
        action_repeat: int = 8,
        seed: int = None,
        env: gym.Env = None  # Optional: pass existing env (for video recording)
    ):
        # Use provided env or create new one
        if env is not None:
            self.env = env
        else:
            self.env = gym.make(env_name, continuous=True)
        
        if seed is not None:
            self.env.reset(seed=seed)
            
        self.img_stack = img_stack
        self.action_repeat = action_repeat
        self.stack = []
        self.av_r = self._reward_memory()
        
    def _rgb2gray(self, rgb: np.ndarray, norm: bool = True) -> np.ndarray:
        """Convert RGB to normalized grayscale"""
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        if norm:
            gray = gray / 128.0 - 1.0  # Normalize to [-1, 1]
        return gray
    
    def _reward_memory(self):
        """Create reward memory for early termination detection"""
        count = 0
        length = 100
        history = np.zeros(length)
        
        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)
        
        return memory
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial stacked state"""
        self.av_r = self._reward_memory()
        img_rgb, _ = self.env.reset()
        img_gray = self._rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action with frame skipping.
        
        Args:
            action: Action in [0, 1] range from Beta distribution
                   Needs to be transformed:
                   - steering: action[0] * 2 - 1 -> [-1, 1]
                   - gas: action[1] -> [0, 1]
                   - brake: action[2] -> [0, 1]
                   
        Returns:
            state: Stacked grayscale frames
            reward: Total reward over action repeat
            done: Episode termination flag
            truncated: Episode truncation flag
            info: Additional information including 'die' flag
        """
        # Transform action from [0, 1] to appropriate ranges
        transformed_action = np.array([
            action[0] * 2 - 1,  # steering: [-1, 1]
            action[1],          # gas: [0, 1]
            action[2]           # brake: [0, 1]
        ])
        
        total_reward = 0
        die = False
        
        for _ in range(self.action_repeat):
            img_rgb, reward, terminated, truncated, info = self.env.step(transformed_action)
            
            # Check for going off track (die state)
            if terminated:
                die = True
                reward += 100  # Don't heavily penalize dying
                
            # Penalize staying on grass (green areas)
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
                
            total_reward += reward
            
            # Check for early termination (no progress)
            done = True if self.av_r(reward) <= -0.1 else False
            
            if done or die or truncated:
                break
                
        # Update frame stack
        img_gray = self._rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        
        return np.array(self.stack), total_reward, done, truncated, {'die': die}
    
    def render(self):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()


# # -------------------------
# # Training Function
# # -------------------------
# def train_ppo(
#     env_name: str = "CarRacing-v3",
#     total_episodes: int = 2000,
#     max_timesteps: int = 1000,
#     img_stack: int = 4,
#     action_repeat: int = 8,
#     buffer_capacity: int = 2000,
#     lr: float = 1e-3,
#     gamma: float = 0.99,
#     gae_lambda: float = 0.95,
#     clip_param: float = 0.1,
#     ppo_epochs: int = 10,
#     batch_size: int = 128,
#     log_interval: int = 10,
#     save_interval: int = 100,
#     seed: int = 0,
#     render: bool = False,
#     save_path: str = "ppo_carracing"
# ):
#     """
#     Train PPO agent on CarRacing-v3.
    
#     Args:
#         env_name: Environment name
#         total_episodes: Total training episodes
#         max_timesteps: Max timesteps per episode
#         img_stack: Number of stacked frames
#         action_repeat: Number of frames to repeat action
#         buffer_capacity: Rollout buffer capacity
#         lr: Learning rate
#         gamma: Discount factor
#         gae_lambda: GAE lambda
#         clip_param: PPO clipping parameter
#         ppo_epochs: PPO update epochs
#         batch_size: Mini-batch size
#         log_interval: Episodes between logging
#         save_interval: Episodes between saving
#         seed: Random seed
#         render: Whether to render
#         save_path: Model save path
#     """
#     # Set seeds
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     # Create environment
#     env = CarRacingWrapper(
#         env_name=env_name,
#         img_stack=img_stack,
#         action_repeat=action_repeat,
#         seed=seed
#     )
    
#     # Create agent
#     agent = PPOAgent(
#         img_stack=img_stack,
#         action_dim=3,
#         lr=lr,
#         gamma=gamma,
#         gae_lambda=gae_lambda,
#         clip_param=clip_param,
#         ppo_epochs=ppo_epochs,
#         batch_size=batch_size,
#         buffer_capacity=buffer_capacity
#     )
    
#     # Training loop
#     running_score = 0
#     best_score = -float('inf')
    
#     print("=" * 60)
#     print("Starting PPO Training on CarRacing-v3")
#     print("=" * 60)
    
#     for episode in range(1, total_episodes + 1):
#         state = env.reset()
#         episode_reward = 0
        
#         for t in range(max_timesteps):
#             # Select action
#             action, action_log_prob, value = agent.select_action(state)
            
#             # Execute action
#             next_state, reward, done, truncated, info = env.step(action)
            
#             if render:
#                 env.render()
            
#             # Store transition
#             agent.store_transition(state, action, action_log_prob, reward, value, done or truncated)
            
#             episode_reward += reward
#             state = next_state
            
#             # Update if buffer is full
#             if len(agent.buffer) >= buffer_capacity:
#                 update_info = agent.update()
#                 if update_info:
#                     pass  # Could log update info here
            
#             if done or truncated or info.get('die', False):
#                 break
        
#         # Update with remaining buffer
#         if len(agent.buffer) > batch_size:
#             agent.update()
        
#         # Update running score
#         running_score = 0.99 * running_score + 0.01 * episode_reward
        
#         # Logging
#         if episode % log_interval == 0:
#             print(f"Episode {episode:5d} | "
#                   f"Score: {episode_reward:8.2f} | "
#                   f"Running Score: {running_score:8.2f} | "
#                   f"Steps: {t+1:4d}")
        
#         # Save best model
#         if running_score > best_score:
#             best_score = running_score
#             agent.save(f"{save_path}_best.pth")
        
#         # Periodic save
#         if episode % save_interval == 0:
#             agent.save(f"{save_path}_ep{episode}.pth")
        
#         # Check if solved (reward threshold is typically 900)
#         if running_score > 900:
#             print(f"\nSolved! Running score: {running_score:.2f}")
#             agent.save(f"{save_path}_solved.pth")
#             break
    
#     env.close()
#     print("\nTraining completed!")
#     return agent


# def evaluate_ppo(
#     model_path: str,
#     env_name: str = "CarRacing-v3",
#     num_episodes: int = 10,
#     img_stack: int = 4,
#     action_repeat: int = 8,
#     render: bool = True,
#     seed: int = 0
# ):
#     """
#     Evaluate trained PPO agent.
#     """
#     # Create environment
#     env = CarRacingWrapper(
#         env_name=env_name,
#         img_stack=img_stack,
#         action_repeat=action_repeat,
#         seed=seed
#     )
    
#     # Create agent and load model
#     agent = PPOAgent(img_stack=img_stack, action_dim=3)
#     agent.load(model_path)
    
#     total_rewards = []
    
#     for episode in range(1, num_episodes + 1):
#         state = env.reset()
#         episode_reward = 0
        
#         for t in range(1000):
#             action, _, _ = agent.select_action(state, deterministic=True)
#             state, reward, done, truncated, info = env.step(action)
            
#             if render:
#                 env.render()
                
#             episode_reward += reward
            
#             if done or truncated or info.get('die', False):
#                 break
        
#         total_rewards.append(episode_reward)
#         print(f"Episode {episode}: Score = {episode_reward:.2f}")
    
#     env.close()
    
#     print(f"\nAverage Score: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
#     return total_rewards


# if __name__ == "__main__":
#     # Train the agent
#     agent = train_ppo(
#         total_episodes=2000,
#         log_interval=10,
#         save_interval=100,
#         render=False,
#         save_path="ppo_carracing"
#     )
