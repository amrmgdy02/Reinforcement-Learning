from tensordict import TensorDict
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.optim as optim
import time
import os

env_name = "LunarLander-v3"
num_envs = 10
envs = gym.make_vec(env_name, num_envs=num_envs, vectorization_mode="sync", continuous=True)

env = gym.make(env_name, render_mode="rgb_array", continuous=True)
trigger = lambda t: True
recording_output_directory = "./checkpoint_videos"
recording_env = RecordVideo(env, recording_output_directory, episode_trigger=trigger)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        input_shape = int(np.array(envs.single_observation_space.shape).prod())
        self.action_dim = envs.single_action_space.shape[0]
        self.action_low = torch.tensor(envs.single_action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_dim), std=0.01)
        )

        self.log_std = nn.Parameter(torch.ones(self.action_dim) * -1.0)  

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def get_critic_value(self, x):
        x_flat = x.view(x.shape[0], -1).float()
        value = self.critic(x_flat)
        # Clamp critic values to prevent extreme outputs
        return torch.clamp(value, -100, 100)

    def get_actor_values(self, x, action=None):
        x_flat = x.view(x.shape[0], -1).float()
        mean = self.actor(x_flat)
        
        # Clamp mean to prevent extreme action predictions
        mean = torch.clamp(mean, -10, 10)
        
        # log_std clamping
        log_std_clamped = torch.clamp(self.log_std, -5, 0.5)  # std between ~0.007 and ~1.65
        std = log_std_clamped.exp().unsqueeze(0).expand_as(mean)
        
        dist = Normal(mean, std)

        if action is None:
            action = dist.rsample()
            low = self.action_low.to(mean.device)
            high = self.action_high.to(mean.device)
            action = torch.clamp(action, low, high)
        else:
            action = action.to(mean.device).float()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

class PPORolloutCollector:
    def __init__(self, agent, envs, num_steps_per_rollout, device, eval_env=None):
        self.agent = agent
        self.envs = envs
        self.num_envs = envs.num_envs
        self.num_steps_per_rollout = num_steps_per_rollout
        self.device = device
        self.eval_env = eval_env
        self.obs_shape = envs.single_observation_space.shape
        self.action_shape = envs.single_action_space.shape
        self.initial_buffer_shape = (num_steps_per_rollout, envs.num_envs)

        obs, _ = envs.reset()
        self.next_obs = torch.Tensor(obs).to(device)
        self.next_done = torch.zeros(num_envs).to(device)
        
    def _create_buffer(self):
        return TensorDict({
            "obs": torch.zeros(self.initial_buffer_shape + self.obs_shape).to(self.device),
            "actions": torch.zeros(self.initial_buffer_shape + self.action_shape).to(self.device),
            "log_probs": torch.zeros(self.initial_buffer_shape).to(self.device),
            "rewards": torch.zeros(self.initial_buffer_shape).to(self.device),
            "dones": torch.zeros(self.initial_buffer_shape).to(self.device),
            "critic_values": torch.zeros(self.initial_buffer_shape).to(self.device),
        })

    def get_next_rollout(self):
        buffer = self._create_buffer()
        next_obs = self.next_obs
        next_done = self.next_done

        for t in range(self.num_steps_per_rollout):
            buffer["obs"][t] = next_obs
            buffer["dones"][t] = next_done

            with torch.no_grad():
                action, log_prob, entropy = self.agent.get_actor_values(next_obs)
                critic_value = self.agent.get_critic_value(next_obs)
                
                # Check for NaN/Inf during rollout
                if not torch.isfinite(action).all():
                    raise ValueError(f"NaN/Inf in action at step {t}")
                if not torch.isfinite(critic_value).all():
                    raise ValueError(f"NaN/Inf in critic_value at step {t}")
                if not torch.isfinite(log_prob).all():
                    raise ValueError(f"NaN/Inf in log_prob at step {t}")

            buffer["actions"][t] = action
            buffer["log_probs"][t] = log_prob
            buffer["critic_values"][t] = critic_value.flatten()

            action_to_step = action.cpu().numpy()
            next_obs, reward, terminations, truncations, infos = envs.step(action_to_step)

            # Clip rewards to prevent extreme values
            reward = np.clip(reward, -10, 10)
            reward = torch.tensor(reward).to(self.device).view(-1)
            buffer["rewards"][t] = reward

            next_done = np.logical_or(terminations, truncations)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

        buffer['next_obs'] = next_obs
        buffer['next_done'] = next_done
        with torch.no_grad():
            buffer['next_value'] = self.agent.get_critic_value(next_obs).reshape(1, -1)
        self.next_obs = next_obs
        self.next_done = next_done
        return buffer

    def run_eval_rollout(self, num_episodes: int = 5):
        assert self.eval_env is not None, "No eval_env provided."
        final_rewards = []
        total_entropies = []

        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            obs = torch.tensor(obs, device=self.device).unsqueeze(0)
            done = False
            episode_rewards = []
            episode_entropies = []

            while not done:
                with torch.no_grad():
                    action, _, entropy = self.agent.get_actor_values(obs)
                    action = action.squeeze(0)
                obs_np, reward, term, trunc, _ = self.eval_env.step(action.cpu().numpy())
                done = term or trunc
                obs = torch.tensor(obs_np, device=self.device).unsqueeze(0)
                episode_rewards.append(float(reward))
                episode_entropies.append(float(entropy.mean().item()) if torch.is_tensor(entropy) else float(entropy))

            final_rewards.append(sum(episode_rewards))
            total_entropies.append(sum(episode_entropies) / len(episode_entropies))

        return {
            "average_reward_per_run": sum(final_rewards) / len(final_rewards),
            "average_entropy_per_run": sum(total_entropies) / len(total_entropies),
        }

def compute_gae(buffer, gamma=0.99, gae_lambda=0.95):
    next_done = buffer['next_done']
    critic_values = buffer['critic_values']
    rewards = buffer['rewards']
    dones = buffer['dones']
    num_steps_per_rollout = len(rewards)
    last_step_critic_values = buffer['next_value']

    advantages = torch.zeros_like(rewards).to(device)
    last_gae_value = 0
    
    with torch.no_grad():
        for t in reversed(range(num_steps_per_rollout)):
            if t == num_steps_per_rollout - 1:
                next_nonterminal = 1.0 - next_done
                next_values = last_step_critic_values
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_values = critic_values[t + 1]

            discounted_estimated_future_reward = gamma * next_values * next_nonterminal
            td_error = rewards[t] + discounted_estimated_future_reward - critic_values[t]
            current_advantage = td_error + gamma * gae_lambda * next_nonterminal * last_gae_value
            
            # Clamp advantages to prevent extreme values
            current_advantage = torch.clamp(current_advantage, -100, 100)
            advantages[t] = current_advantage
            last_gae_value = current_advantage
    
    buffer['advantages'] = advantages
    returns = advantages + critic_values
    buffer['returns'] = returns
    return buffer

# PPO update
def run_ppo_update_step(update_epochs_per_rollout, agent, optimizer, buffer, batch_size, minibatch_size,
                      clip_coef=0.2, ent_coef=0.01, vf_coef=0.5,
                      should_normalize_advantages=True, max_grad_norm=0.5):
    
    b_obs = buffer['obs'].reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = buffer['log_probs'].reshape(-1)
    b_actions = buffer['actions'].reshape((-1,) + envs.single_action_space.shape)
    b_advantages = buffer['advantages'].reshape(-1)
    b_returns = buffer['returns'].reshape(-1)
    
    if not torch.isfinite(b_advantages).all():
        print("Warning: Non-finite advantages detected, skipping update")
        return
    
    b_inds = np.arange(batch_size)

    for epoch in range(update_epochs_per_rollout):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            try:
                _, new_log_prob, entropy = agent.get_actor_values(b_obs[mb_inds], action=b_actions[mb_inds])
                new_critic_value = agent.get_critic_value(b_obs[mb_inds])

                if not torch.isfinite(new_log_prob).all() or not torch.isfinite(new_critic_value).all():
                    print(f"Warning: NaN in forward pass at epoch {epoch}, skipping minibatch")
                    continue

                log_ratio = new_log_prob - b_logprobs[mb_inds]
                # Clamp log_ratio to prevent numerical overflow in exp
                log_ratio = torch.clamp(log_ratio, -20, 20)
                ratio = log_ratio.exp()

                mb_advantages = b_advantages[mb_inds]
                
                if should_normalize_advantages:
                    adv_mean = mb_advantages.mean()
                    adv_std = mb_advantages.std()
                    # normalization safety
                    if torch.isfinite(adv_mean) and torch.isfinite(adv_std) and adv_std > 1e-8:
                        mb_advantages = (mb_advantages - adv_mean) / (adv_std + 1e-8)
                        mb_advantages = torch.clamp(mb_advantages, -10, 10)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_critic_value = new_critic_value.view(-1)
                v_loss = 0.5 * ((new_critic_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                if not torch.isfinite(loss):
                    print(f"Warning: Non-finite loss at epoch {epoch}, skipping minibatch")
                    continue

                optimizer.zero_grad()
                loss.backward()
                
                has_nan_grad = False
                for param in agent.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"Warning: NaN gradients at epoch {epoch}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()
                
            except Exception as e:
                print(f"\nError in PPO update: {e}")
                optimizer.zero_grad()
                continue

# ============= TRAINING SETUP =============
agent = Agent(envs).to(device)
# Lower learning rate for continuous control
learning_rate = 1e-4
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

num_training_iterations = 1000
num_steps_per_rollout = 1000
minibatch_size = 256
update_epochs_per_rollout = 4
iterations_per_eval = 100
runs_per_eval = 5

output_model_checkpoints_base = "./checkpoints"
os.makedirs(output_model_checkpoints_base, exist_ok=True)

rollout_collector = PPORolloutCollector(
    agent=agent,
    envs=envs,
    num_steps_per_rollout=num_steps_per_rollout,
    device=device,
    eval_env=recording_env
)

elapsed_timesteps = 0
start_time = time.time()

for iteration in range(num_training_iterations):
    try:
        agent.train()
        
        print(f"\rtraining iteration: {iteration}/{num_training_iterations} - runtime: {time.time() - start_time:.2f}", end='', flush=True)

        buffer = rollout_collector.get_next_rollout()
        buffer = compute_gae(buffer)

        run_ppo_update_step(
            update_epochs_per_rollout,
            agent,
            optimizer,
            buffer,
            batch_size=num_envs * num_steps_per_rollout,
            minibatch_size=minibatch_size
        )

        elapsed_timesteps += num_envs * num_steps_per_rollout

        if iteration % iterations_per_eval == 0 or iteration == num_training_iterations - 1:
            with torch.no_grad():
                agent.eval()
                eval_stats = rollout_collector.run_eval_rollout(num_episodes=runs_per_eval)
                avg_reward = eval_stats["average_reward_per_run"]
                avg_entropy = eval_stats['average_entropy_per_run']

                ckpt_path = os.path.join(output_model_checkpoints_base, f"checkpoint_{iteration}.pth")
                agent.save(ckpt_path)
                agent.train()

            elapsed = time.time() - start_time
            print(f"\riteration {iteration}/{num_training_iterations} | timesteps: {elapsed_timesteps} | "
                    f"avg_eval_reward: {avg_reward:.2f} | avg_entropy: {avg_entropy:.2f} | runtime: {elapsed:.2f}s")
    
    except (ValueError, RuntimeError) as e:
        print(f"\n\nFatal error at iteration {iteration}: {e}")
        print("Attempting to save emergency checkpoint...")
        emergency_path = os.path.join(output_model_checkpoints_base, f"emergency_checkpoint_{iteration}.pth")
        agent.save(emergency_path)
        print(f"Emergency checkpoint saved to {emergency_path}")
        break

final_ckpt_path = os.path.join(output_model_checkpoints_base, "!final_model.pth")
agent.save(final_ckpt_path)
print("\nTraining complete!")