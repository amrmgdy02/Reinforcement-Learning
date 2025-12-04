import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

# -----------------------------
# Actor-Critic Network
# -----------------------------
"""
this is shared actor-critic network for A2C, SAC, and PPO agents.
It consists of a shared feature extractor followed by separate actor and critic heads.
"""
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = [] 
        dims = [state_dim] + list(hidden_dims) # input dimension followed by hidden layers
        #loop builds
        #Linear(state_dim - 256)
        # ReLU
        # Linear(256 - 256)
        # ReLU
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        self.shared = nn.Sequential(*layers)  #Make these layers run one after another automatically
        
        # Actor and Critic heads
        # Linear(256 - action_dim) for actor output
        # Linear(256 - 1) for critic output
        self.actor = nn.Linear(dims[-1], action_dim)  # policy output
        self.critic = nn.Linear(dims[-1], 1)          # value output

    def forward(self, x):
        """
        Forward pass through the network.
        Returns policy logits and state value.
        param x: input state
        return: policy logits, state value
        """
        x = self.shared(x) 
        policy_logits = self.actor(x)
        value = self.critic(x)
        return policy_logits, value


# -----------------------------
# A2C Agent
# -----------------------------
class A2CAgent:
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256),
                 lr=1e-3, gamma=0.99, entropy_coef=0.01):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.action_dim = action_dim
        
        self.ac_net = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        
    def select_action(self, state):
        """
        Select action based on the current policy.
        param state: current state
        return: selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device) # add batch dimension and convert to tensor and move to device
        with torch.no_grad(): # no gradient calculation for action selection because we don't need to update weights here
            policy_logits, _ = self.ac_net(state) # get policy logits
            probs = F.softmax(policy_logits, dim=-1) # convert logits to probabilities 
            #,dim=-1 means apply softmax along the last dimension which IS the action dimension here            
        action = torch.multinomial(probs, 1).item() # sample action from the probability distribution
        return action

    def compute_returns(self, rewards, dones, last_value=0):
        """
        Compute discounted returns.
        param rewards: list of rewards
        param dones: list of done flags
        param last_value: value of the last state
        return: list of discounted returns
        """
        R = last_value
        returns = []
        # Iterate over rewards and dones in reverse order to compute returns
        for r, done in zip(reversed(rewards), reversed(dones)): 
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def update(self, states, actions, rewards, dones):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(self.compute_returns(rewards, dones)).unsqueeze(1).to(self.device)
        
        logits, values = self.ac_net(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Advantage
        advantages = returns - values
        
        # Actor loss
        action_log_probs = log_probs.gather(1, actions)
        actor_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Entropy loss (encourages exploration)
        entropy_loss = -(probs * log_probs).sum(dim=1).mean()
        
        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def save(self, path):
        torch.save(self.ac_net.state_dict(), path)
        
    def load(self, path):
        self.ac_net.load_state_dict(torch.load(path))


