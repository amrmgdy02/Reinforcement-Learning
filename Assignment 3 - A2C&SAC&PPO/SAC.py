import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple

# -------------------------
# Device
# -------------------------
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
# Actor Networks
# -------------------------

class ContinuousActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.max_action = float(max_action)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std


class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.logits = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.logits(x)


# -------------------------
# Critic Network (Twin Q)
# -------------------------
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete=False):
        super().__init__()
        self.discrete = discrete
        input_dim = state_dim if discrete else state_dim + action_dim

        # Q1
        self.l1 = nn.Linear(input_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(input_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action=None):
        if self.discrete:
            # action is used as index
            q_values = []
            for q_net in [(self.l1, self.l2, self.l3), (self.l4, self.l5, self.l6)]:
                x = F.relu(q_net[0](state))
                x = F.relu(q_net[1](x))
                q_all = q_net[2](x)  # shape = batch x 1 for each action
                q_values.append(q_all)
            return q_values[0], q_values[1]
        else:
            sa = torch.cat([state, action], dim=1)
            # Q1
            x1 = F.relu(self.l1(sa))
            x1 = F.relu(self.l2(x1))
            q1 = self.l3(x1)
            # Q2
            x2 = F.relu(self.l4(sa))
            x2 = F.relu(self.l5(x2))
            q2 = self.l6(x2)
            return q1, q2


# -------------------------
# SAC Agent
# -------------------------
class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.05,
        action_type="continuous",
    ):
        self.device = get_device()
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_type = action_type

        # Actor
        if action_type == "continuous":
            self.actor = ContinuousActor(state_dim, action_dim, max_action).to(self.device)
        else:
            self.actor = DiscreteActor(state_dim, action_dim).to(self.device)

        # Critic
        self.critic = CriticNetwork(state_dim, action_dim, discrete=(action_type == "discrete")).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, discrete=(action_type == "discrete")).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.max_action = max_action
        self._eps = 1e-6

    # -------------------------
    # Action selection
    # -------------------------
    def select_action(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if self.action_type == "continuous":
            mean, log_std = self.actor(state_tensor)
            if deterministic:
                action = torch.tanh(mean) * self.max_action
            else:
                std = log_std.exp()
                z = torch.distributions.Normal(mean, std).rsample()
                action = torch.tanh(z) * self.max_action
            return action.cpu().detach().numpy().flatten()
        else:
            logits = self.actor(state_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            return action.cpu().numpy().item()

    # -------------------------
    # Continuous: sample action & log prob
    # -------------------------
    def _sample_action_and_logprob(self, mean, log_std):
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        action = tanh_z * self.max_action
        log_det_jacobian = torch.log(torch.clamp(1 - tanh_z.pow(2), min=self._eps)).sum(dim=1, keepdim=True)
        log_prob = normal.log_prob(z).sum(dim=1, keepdim=True) - log_det_jacobian
        return action, log_prob

    # -------------------------
    # Training step
    # -------------------------
    def train(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        if len(replay_buffer) < batch_size:
            return 0, 0

        states_np, actions_np, rewards_np, next_states_np, dones_np = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_np).to(self.device)
        not_dones = (1.0 - torch.FloatTensor(dones_np).unsqueeze(1)).to(self.device)

        # -------------------------
        # Critic Update
        # -------------------------
        with torch.no_grad():
            if self.action_type == "continuous":
                next_mean, next_log_std = self.actor(next_states)
                next_action, next_log_prob = self._sample_action_and_logprob(next_mean, next_log_std)
                target_q1, target_q2 = self.critic_target(next_states, next_action)
                min_q = torch.min(target_q1, target_q2)
                target_q = rewards + not_dones * self.gamma * (min_q - self.alpha * next_log_prob)
                actions = torch.FloatTensor(actions_np).to(self.device)
            else:
                logits = self.actor(next_states)
                dist = torch.distributions.Categorical(logits=logits)
                next_action = dist.sample()
                log_prob = dist.log_prob(next_action).unsqueeze(1)
                target_q1, target_q2 = self.critic_target(next_states)
                min_q = torch.min(target_q1, target_q2)
                target_q = rewards + not_dones * self.gamma * (min_q - self.alpha * log_prob)
                actions = torch.LongTensor(actions_np).to(self.device)

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------------
        # Actor Update
        # -------------------------
        if self.action_type == "continuous":
            mean, log_std = self.actor(states)
            action_new, log_prob_new = self._sample_action_and_logprob(mean, log_std)
            q1_new, q2_new = self.critic(states, action_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob_new - q_new).mean()
        else:
            logits = self.actor(states)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_action = dist.sample()
            log_prob = dist.log_prob(sampled_action).unsqueeze(1)
            q1_new, q2_new = self.critic(states)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------------------------
        # Soft update target critic
        # -------------------------
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic.state_dict(), filename + "_critic.pth")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + "_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target.pth", map_location=self.device))
