import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import gymnasium as gym
import wandb
import os
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame):
    """Preprocess CarRacing frame: grayscale and normalize"""
    # Convert RGB to grayscale
    gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    # Normalize to [0, 1]
    normalized = gray / 255.0
    return normalized.astype(np.float32)


class FrameStack:
    """Stack consecutive frames for temporal information"""

    def __init__(self, num_frames=4):
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

    def reset(self, frame):
        processed = preprocess_frame(frame)
        for _ in range(self.num_frames):
            self.frames.append(processed)
        return self.get_state()

    def update(self, frame):
        processed = preprocess_frame(frame)
        self.frames.append(processed)
        return self.get_state()

    def get_state(self):
        return np.stack(self.frames, axis=0)  # Shape: (num_frames, H, W)


class ReplayBuffer(object):
    def __init__(self, state_shape, action_dim, max_size=50000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

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


class CNNActor(nn.Module):
    def __init__(self, num_frames, action_dim, max_action):
        super(CNNActor, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # conv output assumptions for 96x96 input
        conv_output_size = 64 * 8 * 8

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class CNNCritic(nn.Module):
    def __init__(self, num_frames, action_dim):
        super(CNNCritic, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_output_size = 64 * 8 * 8

        # Fully connected layers (state + action)
        self.fc1 = nn.Linear(conv_output_size + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten

        # Concatenate with action
        x = torch.cat([x, action], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TD3_CNN(object):
    def __init__(
        self,
        num_frames,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=1e-4,
        critics_lrs=(1e-4, 1e-4),
    ):
        self.actor = CNNActor(num_frames, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = CNNCritic(num_frames, action_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critics_lrs[0])

        self.critic_2 = CNNCritic(num_frames, action_dim).to(device)
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
        # state shape: (num_frames, H, W)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dimension
        action = self.actor(state).cpu().detach().numpy().flatten()
        return np.clip(action, -self.max_action, self.max_action)

    def _train_one_batch(self, replay_buffer, batch_size=128):
        self.total_iters += 1

        s, a, s_next, r, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action = self.actor_target(s_next)
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

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
        self,
        env,
        replay_buffer,
        total_timesteps,
        batch_size=128,
        steps_to_log=5000,
        start_timesteps=10000,
        exploration_noise=0.1,
        num_frames=4,
        save_every=10_000,  # save checkpoint every save_every steps
    ):
        frame_stack = FrameStack(num_frames)

        episodes_rewards = []
        episode_reward = 0
        episode_duration = 0

        frame, _ = env.reset()
        state = frame_stack.reset(frame)

        for t in range(total_timesteps):

            if replay_buffer.size < start_timesteps:
                action = env.action_space.sample()
            else:
                action = self.select_action(state)
                noise = np.random.normal(0, exploration_noise * self.max_action, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)

            next_frame, reward, done, truncated, _ = env.step(action)
            next_state = frame_stack.update(next_frame)

            replay_buffer.add(state, action, next_state, reward, done or truncated)
            state = next_state

            if replay_buffer.size >= batch_size and replay_buffer.size >= start_timesteps:
                self._train_one_batch(replay_buffer, batch_size)

            episode_reward += reward
            episode_duration += 1

            # Save model checkpoint every `save_every` steps
            if (t + 1) % save_every == 0:
                os.makedirs("TD3-CNN-models", exist_ok=True)
                ckpt_name = f"TD3-CNN-models/td3_cnn_step_{t+1}"
                self.save(ckpt_name)
                print(f"[Step {t+1}] Model checkpoint saved to: {ckpt_name}_*.pth")
                # optionally log a artifact to wandb (uncomment if you want artifact logging):
                # wandb.save(ckpt_name + "_actor.pth")
                # wandb.save(ckpt_name + "_critic_1.pth")
                # wandb.save(ckpt_name + "_critic_2.pth")

            if done or truncated:
                print(
                    f"Step: {t+1} | Episode: {len(episodes_rewards)+1} | "
                    f"Reward: {episode_reward:.1f} | Duration: {episode_duration}"
                )
                episodes_rewards.append(episode_reward)

                episode_idx = len(episodes_rewards)
                wandb.log({"training_episode_reward": episode_reward}, step=episode_idx)

                episode_reward = 0
                episode_duration = 0
                frame, _ = env.reset()
                state = frame_stack.reset(frame)

            if (t + 1) % steps_to_log == 0:
                avg = np.mean(episodes_rewards[-10:]) if len(episodes_rewards) >= 10 else np.mean(episodes_rewards)
                std = np.std(episodes_rewards[-10:]) if len(episodes_rewards) >= 1 else 0.0
                print(f"[{t+1}] Average reward (last 10 episodes): {avg:.2f}")
                print(f"STD of reward (last 10 episodes): {std:.2f}")

                wandb.log(
                    {
                        "avg_reward_last_10": avg,
                        "std_reward_over_10_episodes": std,
                        "timestep": t + 1,
                    }
                )

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_1.state_dict(), filename + "_critic_1.pth")
        torch.save(self.critic_2.state_dict(), filename + "_critic_2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic_1.pth"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic_2.pth"))

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)


if __name__ == "__main__":
    env_name = "CarRacing-v3"
    num_frames = 4

    wandb.login()

    run = wandb.init(
        project="Final_TD3-CNN-CarRacing-v3",
        name=f"TD3_CNN_{env_name}_Train_Eval",
        config={
            "environment": env_name,
            "algorithm": "TD3-CNN",
            "num_frames": num_frames,
            "exploration_noise": 0.1,
            "total_timesteps": 500_000,
            "start_timesteps": 10_000,
            "batch_size": 128,
            "discount": 0.99,
            "tau": 0.005,
            "policy_noise": 0.2,
            "noise_clip": 0.5,
            "policy_freq": 5,
            "actor_lr": 1e-4,
            "critic_lr": 1e-4,
            "buffer_size": 50_000,
        },
    )

    env = gym.make(env_name, render_mode=None)

    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"\n{'='*60}")
    print(f"Training TD3 with CNN for {env_name}")
    print(f"Action dim: {action_dim}, Max action: {max_action}")
    print(f"State: {num_frames} stacked grayscale frames (96x96)")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    agent = TD3_CNN(
        num_frames=num_frames,
        action_dim=action_dim,
        max_action=max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=5,
        actor_lr=1e-4,
        critics_lrs=(1e-4, 1e-4),
    )

    state_shape = (num_frames, 96, 96)  # CarRacing frames are 96x96
    buffer = ReplayBuffer(state_shape, action_dim, max_size=50_000)

    agent.train(
        env,
        buffer,
        total_timesteps=500_000,
        batch_size=128,
        steps_to_log=5000,
        start_timesteps=25_000,
        exploration_noise=0.2,
        num_frames=num_frames,
        save_every=10_000,  # checkpoint frequency
    )

    # Evaluation
    print("\n" + "=" * 60)
    print("Starting Evaluation...")
    print("=" * 60 + "\n")

    eval_episodes = 50
    eval_rewards = []
    frame_stack_eval = FrameStack(num_frames)

    for ep in range(eval_episodes):
        frame, _ = env.reset()
        state = frame_stack_eval.reset(frame)
        done = False
        truncated = False
        ep_reward = 0

        while not (done or truncated):
            action = agent.select_action(state)
            next_frame, reward, done, truncated, _ = env.step(action)
            state = frame_stack_eval.update(next_frame)
            ep_reward += reward

        eval_rewards.append(ep_reward)
        print(f"Evaluation Episode {ep+1}: Reward: {ep_reward:.2f}")

        wandb.log({"eval_episode_reward": ep_reward})

    avg_eval_reward = np.mean(eval_rewards)
    std_eval_reward = np.std(eval_rewards)
    print(f"\n{'='*60}")
    print(f"Average Evaluation Reward over {eval_episodes} episodes: {avg_eval_reward:.2f}")
    print(f"STD Evaluation over {eval_episodes} episodes: {std_eval_reward:.2f}")
    print(f"{'='*60}\n")

    wandb.log({"eval_avg_reward": avg_eval_reward, "eval_std_reward": std_eval_reward})

    # Save the final agent
    os.makedirs("TD3-CNN-models", exist_ok=True)
    agent.save(f"TD3-CNN-models/td3_cnn_{env_name}")

    # Save the configs
    with open(f"TD3-CNN-models/td3_cnn_{env_name}_config.txt", "w") as f:
        for key, value in run.config.items():
            f.write(f"{key}: {value}\n")

    env.close()
    wandb.finish()

    print(f"\n{'='*60}")
    print("Training and Evaluation completed!")
    print(f"Models saved to: TD3-CNN-models/td3_cnn_{env_name}")
    print(f"{'='*60}")
