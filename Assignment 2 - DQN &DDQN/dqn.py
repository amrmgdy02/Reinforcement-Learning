import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from abc import ABC, abstractmethod

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

    """
        Sample a batch of transitions.
        what it do in detail:
        1. Randomly sample 'batch_size' transitions from the buffer.
        2. Unzip the sampled transitions into separate lists for states, actions, rewards, next_states, and dones.
        3. Convert these lists into numpy arrays and ensure they have the correct data types.
        4. Return the arrays as a tuple.
        Returns:
            A tuple of (states, actions, rewards, next_states, dones)
    """
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size) #random.sample will return a list of Transition based on batch_size
        tuples = [(t.state, t.action, t.reward, t.next_state, t.done) for t in batch] # unzip the batch into separate lists
        states, actions, rewards, next_states, dones = map(np.array, zip(*tuples)) #convert lists to numpy arrays
        return (
            states,
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_states,
            dones.astype(np.uint8),
        )

    def __len__(self):
        return len(self.buffer)

class BaseAgent(ABC):
    """
    Base class for DQN and DDQN agents.
    what it do in detail:
    1. Initializes the agent with state and action dimensions, hidden layer sizes, optimizer type, loss function, learning rate, and discount factor.
    2. Sets up the device for computation (GPU if available, otherwise CPU).
    3. Creates a replay buffer to store experiences.
    4. Defines abstract methods for optimizing the network, saving, and loading model parameters, which must be implemented by subclasses.
    5. Provides methods for resetting memory, storing transitions, and selecting actions using an epsilon
    -greedy policy.
    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        hidden_dims (tuple): Sizes of hidden layers.
        optimizer (str): Optimizer type ("adam" or "sgd").
        loss_fn (callable): Loss function.
        lr (float): Learning rate.
        gamma (float): Discount factor.
        
    """
    def __init__(self, 
                 state_dim, action_dim, hidden_dims=(256, 256),
                 optimizer="adam", loss_fn=F.mse_loss, lr=1e-3, gamma=0.99):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.loss_fn = loss_fn

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
        # Define network in subclass
        self.policy_network = None
        self.optimizer = None

    def reset_memory(self):
        self.replay_buffer = ReplayBuffer(capacity=10000)
        
    def store_transition(self, state, action, reward, next_state, done):
        transition = Transition(state, action, reward, next_state, done)
        self.replay_buffer.push(transition)

    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_network(state)
        return q_values.argmax().item()

    @abstractmethod
    def optimize_step(self, batch_size: int):
        """Train the network (to be implemented by subclass)."""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save the model parameters."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load the model parameters."""
        pass

# -----------------------------
# DQN network
# -----------------------------
class DQN(nn.Module):
    """
    Simple feedforward neural network for DQN.
    what it do in detail:
    1. Initializes a feedforward neural network with specified input and output dimensions, and hidden layer sizes.
    2. Constructs the network architecture using linear layers followed by ReLU activation functions.
    Args:
        input_dim (int): Dimension of the input state.
        output_dim (int): Dimension of the output action space.
        hidden_dims (tuple): Sizes of hidden layers.
        
    """
    def __init__(self, input_dim, output_dim, hidden_dims=(256, 256)):
        super(DQN, self).__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# -----------------------------
# DQNAgent (inherits BaseAgent)
# -----------------------------
class DQNAgent(BaseAgent):
    def __init__(self, 
                 state_dim, action_dim, hidden_dims=(256, 256),
                 optimizer="adam", loss_fn=F.mse_loss, lr=1e-3, gamma=0.99):
        
        super().__init__(state_dim, action_dim, hidden_dims, optimizer, loss_fn, lr, gamma)
        
        # single Q-network
        self.policy_network = DQN(state_dim, action_dim, hidden_dims).to(self.device) 
        
        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

    def optimize_step(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.policy_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath: str):
        torch.save(self.policy_network.state_dict(), filepath)
        
    def load(self, filepath: str):
        self.policy_network.load_state_dict(torch.load(filepath))

# -----------------------------
# DDQNAgent (inherits BaseAgent)
# -----------------------------
class DDQNAgent(BaseAgent):
    def __init__(self, 
                 state_dim, action_dim, hidden_dims=(256, 256),
                 optimizer="adam", loss_fn=F.mse_loss, lr=1e-3, gamma=0.99,
                 target_update_freq=100):
        
        super().__init__(state_dim, action_dim, hidden_dims, optimizer, loss_fn, lr, gamma)

        # Two networks
        self.policy_network = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.policy_network.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        
        self.target_update_freq = target_update_freq
        self.train_step = 0

    def optimize_step(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        q_values = self.policy_network(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            
    def save(self, filepath: str):
        torch.save(self.policy_network.state_dict(), filepath + "_policy")
        torch.save(self.target_network.state_dict(), filepath + "_target")
        
    def load(self, filepath: str):
        self.policy_network.load_state_dict(torch.load(filepath + "_policy"))
        self.target_network.load_state_dict(torch.load(filepath + "_target"))