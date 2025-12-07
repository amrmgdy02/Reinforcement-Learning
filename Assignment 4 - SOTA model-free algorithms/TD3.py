import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gymnasium as gym
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(s):
    return np.array(s, dtype=np.float32).reshape(-1)

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=256_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[indices]).to(self.device),
            torch.FloatTensor(self.action[indices]).to(self.device),
            torch.FloatTensor(self.next_state[indices]).to(self.device),
            torch.FloatTensor(self.reward[indices]).to(self.device),
            torch.FloatTensor(self.not_done[indices]).to(self.device),
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class TD3(object):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action,
        discount=0.99, 
        tau=0.005,
        policy_noise=0.2, 
        noise_clip=0.5, 
        policy_freq=2,
        actor_lr=3e-4, 
        critics_lrs=(3e-4, 3e-4)
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critics_lrs[0])

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critics_lrs[1])

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq

        self.total_iters = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().detach().numpy().flatten()
        return np.clip(action, -self.max_action, self.max_action)

    def _train_one_batch(self, replay_buffer, batch_size=256):
        self.total_iters += 1

        s, a, s_next, r, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(s_next)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (next_action + noise).clamp(
                -self.max_action, self.max_action
            )

            target_q1 = self.critic_1_target(s_next, next_action)
            target_q2 = self.critic_2_target(s_next, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = r + not_done * self.discount * target_q

        current_q1 = self.critic_1(s, a)
        critic_1_loss = F.mse_loss(current_q1, target_q)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        current_q2 = self.critic_2(s, a)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        if self.total_iters % self.policy_freq == 0:
            actor_loss = -self.critic_1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft updates
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(
        self, env, replay_buffer, total_timesteps,
        batch_size=256, steps_to_log=10000,
        start_timesteps=20000, exploration_noise=0.1, noise_label="0.1"
    ):
        state, _ = env.reset()
        state = flatten_state(state)

        episodes_rewards = []
        episode_reward = 0
        episode_duration = 0

        s, _ = env.reset()
        s = flatten_state(s)

        for t in range(total_timesteps):
            if replay_buffer.size < start_timesteps:
                action = env.action_space.sample()
            else:
                action = self.select_action(np.array(state))
                noise = np.random.normal(0, exploration_noise * self.max_action, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)

            next_state, reward, done, truncated, _ = env.step(action)
            next_state = flatten_state(next_state)
            replay_buffer.add(state, action, next_state, reward, done or truncated)
            state = next_state

            if replay_buffer.size >= batch_size and replay_buffer.size >= start_timesteps:
                self._train_one_batch(replay_buffer, batch_size)

            episode_reward += reward
            episode_duration += 1

            if done or truncated:
                print(
                    f"Noise {noise_label} | Step: {t+1} | Episode: {len(episodes_rewards)+1} | "
                    f"Reward: {episode_reward:.1f} | Duration: {episode_duration}"
                )
                episodes_rewards.append(episode_reward)
                
                # Log to W&B with noise identifier
                wandb.log({
                    f"episode_reward_exploration_noise_{noise_label}": episode_reward,
                    f"episode_duration_exploration_noise_{noise_label}": episode_duration,
                    "timestep": t + 1
                })
                
                episode_reward = 0
                episode_duration = 0
                state, _ = env.reset()
                state = flatten_state(state)

            if (t + 1) % steps_to_log == 0:
                avg = np.mean(episodes_rewards[-10:]) if len(episodes_rewards) >= 10 else np.mean(episodes_rewards)
                print(f"Noise {noise_label} | [{t+1}] Average reward (last 10 episodes): {avg:.2f}")
                
                # Log average reward to W&B with noise identifier
                wandb.log({
                    f"avg_reward_last_10_noise_{noise_label}": avg,
                    "timestep": t + 1
                })


if __name__ == "__main__":
    env_name = "LunarLander-v3"
    exploration_noises = [0.05 , 0.1, 0.3, 0.5]
    
    wandb.login()
    
    run = wandb.init(
        project="TD3-LunarLander-v3-Exploration-Noise-Tuning",
        name=f"TD3_{env_name}_all_noises",
        config={
            "environment": env_name,
            "algorithm": "TD3",
            "exploration_noises": exploration_noises,
            "total_timesteps": 50_000,
            "start_timesteps": 10_000,
            "batch_size": 256,
            "discount": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 2,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4
        }
    )
    
    env = gym.make(env_name, continuous=True)
    sample_obs, _ = env.reset()
    flat_dim = flatten_state(sample_obs).shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    for noise in exploration_noises:
        print(f"\n{'='*60}")
        print(f"Training with exploration noise: {noise}")
        print(f"{'='*60}\n")
        
        agent = TD3(flat_dim, action_dim, max_action)
        
        if env_name == "CarRacing-v3":
            buffer = ReplayBuffer(flat_dim, action_dim, max_size=1_000)
        else:
            buffer = ReplayBuffer(flat_dim, action_dim)

        agent.train(
            env,
            buffer,
            total_timesteps=30_000,
            start_timesteps=5_000,
            exploration_noise=noise,
            noise_label=str(noise)
        )

    env.close()
    wandb.finish()
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")
