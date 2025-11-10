import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class Transition(object):
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
        self._idx = 0

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            states,
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_states,
            dones.astype(np.uint8),
        )

    def __len__(self):
        return len(self.buffer)
    
    
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 128)):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# Class to deal with DQN Agent (only one nn)
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=(128, 128), optimizer=torch.optim.Adam, loss=F.mse_loss, lr=1e-3, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network: DQN = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optimizer(self.q_network.parameters(), lr=lr)
        self.loss = loss
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
    def reset_memory(self):
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_network.net[-1].out_features - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def train(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.q_network(next_states).max(1)[0].unsqueeze(1)
            #target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            target_q_values = rewards + self.gamma * next_q_values # which is more accurate??

        loss = self.loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()