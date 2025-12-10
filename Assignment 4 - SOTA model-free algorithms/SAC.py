import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Transition & ReplayBuffer
# -------------------------
class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.capacity = capacity
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        random.seed(seed)
        np.random.seed(seed)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        """
        Returns: states, actions, rewards, next_states, dones (all numpy arrays)
        """
        if len(self.buffer) < batch_size:
            raise ValueError("Buffer size less than batch size.")
            
        batch = random.sample(self.buffer, batch_size)
        tuples = [(t.state, t.action, t.reward, t.next_state, t.done) for t in batch]
        states, actions, rewards, next_states, dones = map(np.array, zip(*tuples))

        return (
            states.astype(np.float32),
            actions.astype(np.float32),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


# -------------------------
# CNN Feature Extractor for Image-based Environments
# -------------------------
class CNNFeatureExtractor(nn.Module):
    """CNN for extracting features from image observations (e.g., CarRacing)"""
    def __init__(self, input_channels=3, feature_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size after convolutions (for 96x96 input)
        # After conv1: (96 - 8) / 4 + 1 = 23
        # After conv2: (23 - 4) / 2 + 1 = 10
        # After conv3: (10 - 3) / 1 + 1 = 8
        # So output is 64 * 8 * 8 = 4096 where 64 is the number of output channels from conv3
        
        #Flatten all feature maps into 1D vector
        #feed into fully connected layer to get desired feature_dim which is 256 by default
        self.fc = nn.Linear(64 * 8 * 8, feature_dim)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))
        return x


# -------------------------
# Actor Networks
# -------------------------
class ActorNetwork(nn.Module):
    """Standard MLP Actor for vector observations (e.g., LunarLander)"""
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = float(max_action)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20.0, 2.0) 
        return mean, log_std


class CNNActorNetwork(nn.Module):
    """CNN-based Actor for image observations (e.g., CarRacing)"""
    def __init__(self, input_channels, action_dim, max_action, feature_dim=256):
        super(CNNActorNetwork, self).__init__()
        
        self.feature_extractor = CNNFeatureExtractor(input_channels, feature_dim)
        
        self.l1 = nn.Linear(feature_dim, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = float(max_action)

    def forward(self, state):
        # state shape: (batch, channels, height, width)
        features = self.feature_extractor(state)
        x = F.relu(self.l1(features))
        mean = self.mean(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std


# -------------------------
# Critic Networks
# -------------------------
class CriticNetwork(nn.Module):
    """Standard MLP Critic for vector observations (e.g., LunarLander)"""
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        input_dim = state_dim + action_dim
        
        # Q1 definition
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 definition
        self.l4 = nn.Linear(input_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q1 = self.l3(x1)
        
        # Q2 forward pass
        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        q2 = self.l6(x2)
        return q1, q2


class CNNCriticNetwork(nn.Module):
    """CNN-based Critic for image observations (e.g., CarRacing)"""
    def __init__(self, input_channels, action_dim, feature_dim=256):
        super(CNNCriticNetwork, self).__init__()
        
        self.feature_extractor = CNNFeatureExtractor(input_channels, feature_dim)
        
        input_dim = feature_dim + action_dim
        
        # Q1 definition
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        
        # Q2 definition
        self.l4 = nn.Linear(input_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        # state shape: (batch, channels, height, width)
        features = self.feature_extractor(state)
        sa = torch.cat([features, action], dim=1)
        
        # Q1 forward pass
        x1 = F.relu(self.l1(sa))
        x1 = F.relu(self.l2(x1))
        q1 = self.l3(x1)
        
        # Q2 forward pass
        x2 = F.relu(self.l4(sa))
        x2 = F.relu(self.l5(x2))
        q2 = self.l6(x2)
        return q1, q2


# -------------------------
# SAC Agent
# -------------------------
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.05, device=None, use_cnn=False, input_channels=3):
        """
        Args:
            state_dim: For MLP mode, this is the flattened state dimension
                       For CNN mode, this is not used (we use input_channels instead)
            action_dim: Action space dimension
            max_action: Maximum action value
            use_cnn: If True, use CNN networks for image inputs
            input_channels: Number of input channels for CNN (e.g., 3 for RGB)
        """
        
        self.device = device or get_device() 
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.use_cnn = use_cnn

        # Initialize networks based on mode
        if use_cnn:
            print("Initializing SAC with CNN networks for image-based observations")
            self.actor = CNNActorNetwork(input_channels, action_dim, max_action).to(self.device)
            self.critic = CNNCriticNetwork(input_channels, action_dim).to(self.device)
            self.critic_target = CNNCriticNetwork(input_channels, action_dim).to(self.device)
        else:
            print("Initializing SAC with MLP networks for vector observations")
            self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
            self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
            self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self._eps = 1e-6
        self.log_std_min = -20 
        self.log_std_max = 2

    def _prepare_state_tensor(self, state):
        """Convert state to appropriate tensor format"""
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.float().to(self.device)
        else:
            raise ValueError(f"Unsupported state type: {type(state)}")
        
        if self.use_cnn:
            # For CNN: expect shape (batch, channels, height, width)
            # If single sample without batch dim, add it
            if state_tensor.dim() == 3:  # (channels, height, width)
                state_tensor = state_tensor.unsqueeze(0)
        else:
            # For MLP: expect shape (batch, state_dim)
            if state_tensor.dim() == 1:  # (state_dim,)
                state_tensor = state_tensor.unsqueeze(0)
        
        return state_tensor

    def _sample_action_and_logprob(self, mean, log_std) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given mean and log_std -> sample via reparameterization and compute
        log probability corrected for tanh squashing.
        """
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        z = normal.rsample()  
        tanh_z = torch.tanh(z)
        action = tanh_z * self.actor.max_action
        
        log_det_jacobian = torch.log(torch.clamp(1.0 - tanh_z.pow(2), min=self._eps)).sum(dim=1, keepdim=True)
        log_prob = normal.log_prob(z).sum(dim=1, keepdim=True) - log_det_jacobian
        
        return action, log_prob

    def select_action(self, state, deterministic: bool = False):
        """
        Processes a single state and returns an action.
        """
        with torch.no_grad():
            state_tensor = self._prepare_state_tensor(state)
            mean, log_std = self.actor(state_tensor)
            
            if deterministic:
                a = torch.tanh(mean) * self.actor.max_action
            else:
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                z = normal.rsample()
                a = torch.tanh(z) * self.actor.max_action
                
            return a.cpu().numpy().flatten() 

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        if len(replay_buffer) < batch_size:
            return 

        # Sample from buffer
        states_np, actions_np, rewards_np, next_states_np, dones_np = replay_buffer.sample(batch_size)

        # Convert to tensors
        if self.use_cnn:
            # For CNN: reshape to (batch, channels, height, width)
            # Assuming states_np is (batch, channels*height*width)
            # We need to know the original shape - assuming 96x96x3 for CarRacing
            batch_size_actual = states_np.shape[0]
            states = torch.FloatTensor(states_np).reshape(batch_size_actual, 3, 96, 96).to(self.device)
            next_states = torch.FloatTensor(next_states_np).reshape(batch_size_actual, 3, 96, 96).to(self.device)
        else:
            states = torch.FloatTensor(states_np).to(self.device)
            next_states = torch.FloatTensor(next_states_np).to(self.device)
        
        actions = torch.FloatTensor(actions_np).to(self.device) 
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1).to(self.device)
        not_dones = (1.0 - torch.FloatTensor(dones_np).unsqueeze(1)).to(self.device) 

        # 1. Critic Update
        with torch.no_grad():
            next_mean, next_log_std = self.actor(next_states)
            next_action, next_log_prob = self._sample_action_and_logprob(next_mean, next_log_std) 
            
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            
            target_value = min_target_q - self.alpha * next_log_prob
            target_q = rewards + not_dones * self.gamma * target_value

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor Update
        mean, log_std = self.actor(states)
        action_new, log_prob_new = self._sample_action_and_logprob(mean, log_std)
        
        q1_new, q2_new = self.critic(states, action_new) 
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob_new - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft update
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target.pth", map_location=self.device))