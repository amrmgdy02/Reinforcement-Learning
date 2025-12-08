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


# Actor network
class ActorNetwork(nn.Module):
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
        # Clamping log_std for stability
        log_std = torch.clamp(log_std, -20.0, 2.0) 
        return mean, log_std


# Critic (twin Q) network
class CriticNetwork(nn.Module):
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

# SAC Agent
class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.05, device=None):
        
        self.device = device or get_device() 
        self.gamma = gamma
        self.tau = tau # soft update rate for target network for update target network weights smoothly instead of change many weights suddenly
        self.alpha = alpha # entropy coefficient for exploration-exploitation trade-off
        #as alpha increases, exploration increases

        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict()) # Initialize target with critic weights

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr) #this optimizer updates actor parameters
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Small constant for numerical stability (min value for log input)
        self._eps = 1e-6  #when computing log, we use clamp to avoid log(0)
        # A constant for log probability calculation (often used instead of clamping)
        self.log_std_min = -20 
        self.log_std_max = 2

    def _sample_action_and_logprob(self, mean, log_std) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given mean and log_std -> sample via reparameterization and compute
        log probability corrected for tanh squashing.
        
        Returns: action (scaled/squashed), log_prob (sum over action dims)
        """
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        
        # Reparameterization trick: z = mean + std * epsilon
        z = normal.rsample()  
        
        # Squashing: action = tanh(z) * max_action
        tanh_z = torch.tanh(z) #squash to -1 to 1
        action = tanh_z * self.actor.max_action #multiply by max_action to scale to action space
        
        # Log Probability (corrected for change of variables)
        # log_prob_a = log_prob_z - log(|det(d(tanh(z))/dz)|)
        
        # Log of (1 - tanh(z)^2) is the correction term.
        # Use clamp to prevent log(0) which causes NaN or Inf, which is 
        # the most common cause of non-learning in SAC.
        log_det_jacobian = torch.log(torch.clamp(1.0 - tanh_z.pow(2), min=self._eps)).sum(dim=1, keepdim=True)
        
        log_prob = normal.log_prob(z).sum(dim=1, keepdim=True) - log_det_jacobian
        
        return action, log_prob

    def select_action(self, state, deterministic: bool = False):
        """
        Processes a single state and returns an action.
        if deterministic=True, returns the mean action for evaluation.
        else samples from the policy for exploration.
        """
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.float().unsqueeze(0).to(self.device)
                
            mean, log_std = self.actor(state_tensor)
            
            if deterministic:
                # Deterministic (mean) action for evaluation
                a = torch.tanh(mean) * self.actor.max_action
            else:
                # Stochastic (sampled) action for exploration
                std = log_std.exp()
                normal = torch.distributions.Normal(mean, std)
                z = normal.rsample() # Reparameterization trick for sampling this important for backpropagation
                a = torch.tanh(z) * self.actor.max_action
                
            return a.cpu().numpy().flatten() 

    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        if len(replay_buffer) < batch_size:
            return 

        # Retrieve and convert data
        states_np, actions_np, rewards_np, next_states_np, dones_np = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states_np).to(self.device) 
        actions = torch.FloatTensor(actions_np).to(self.device) 
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device) 
        # Convert 0/1 to boolean for masking
        not_dones = (1.0 - torch.FloatTensor(dones_np).unsqueeze(1)).to(self.device) 

        # 1. Critic Update (Target Calculation)
        with torch.no_grad():
            # Sample next action (a') from the current policy for the target value
            next_mean, next_log_std = self.actor(next_states)
            next_action, next_log_prob = self._sample_action_and_logprob(next_mean, next_log_std) 
            
            # Use Target Critic Networks (Q_target) and the Twin Q-trick (min)
            target_q1, target_q2 = self.critic_target(next_states, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            
            # SAC Bellman target: Q_target = R + gamma * (min_Q_target - alpha * log_pi)
            target_value = min_target_q - self.alpha * next_log_prob
            target_q = rewards + not_dones * self.gamma * target_value

        # Calculate current Q-values
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic Loss: MSE between current Q and target Q for both Q1 and Q2
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor Update
        # Sample a new action (a) and its log_prob from the policy for the current state (s)
        mean, log_std = self.actor(states)
        action_new, log_prob_new = self._sample_action_and_logprob(mean, log_std)
        
        # Use CURRENT Critic Networks (Q) to estimate Q for the new action
        q1_new, q2_new = self.critic(states, action_new) 
        q_new = torch.min(q1_new, q2_new)

        # Actor loss (minimize: alpha * log_prob - Q) 
        # This is equivalent to maximizing: Q - alpha * log_prob (max Q + Entropy)
        actor_loss = (self.alpha * log_prob_new - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft update for target critic
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Return losses for monitoring
        return critic_loss.item(), actor_loss.item()


    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target.pth", map_location=self.device))