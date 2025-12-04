"""
    PPO class (extended to support discrete and continuous action spaces).
    Original continuous-only implementation adapted to also handle Discrete action spaces.
"""

import gymnasium as gym
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal, Independent, Categorical

class PPO:
    """
        PPO supporting both continuous (Box) and discrete (Discrete) action spaces.
    """
    def __init__(self, env, **hyperparameters):
        """
        PPO agent that automatically builds actor/critic inside __init__
        and supports both discrete + continuous action spaces.

        env: gymnasium environment (Box observation space + Box or Discrete action space)
        """
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, (gym.spaces.Box, gym.spaces.Discrete))

        self._init_hyperparameters(hyperparameters)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]

        if isinstance(env.action_space, gym.spaces.Box):
            self.discrete = False
            self.act_dim = env.action_space.shape[0]
        else:
            self.discrete = True
            self.act_dim = env.action_space.n

        def mlp(sizes, activation=nn.Tanh(), output_activation=nn.Identity()):
            layers = []
            for i in range(len(sizes) - 1):
                act = activation if i < len(sizes) - 2 else output_activation
                layers.append(nn.Linear(sizes[i], sizes[i+1]))
                layers.append(act)
            return nn.Sequential(*layers)

        actor_output_dim = self.act_dim
        self.actor = mlp(
            sizes=[self.obs_dim, 64, 64, actor_output_dim],
            activation=nn.Tanh(),
            output_activation=nn.Identity()
        )

        self.critic = mlp(
            sizes=[self.obs_dim, 64, 64, 1],
            activation=nn.Tanh(),
            output_activation=nn.Identity()
        )

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        if not self.discrete:
            self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, dtype=torch.float)
        else:
            self.cov_var = None

        # Logger
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,
            'i_so_far': 0,
            'batch_lens': [],
            'batch_rews': [],
            'actor_losses': [],
        }

    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            # Advantage: A = RTG - V(s)
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # actor update
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # critic update
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())

            # Logging and save
            self._log_summary()
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                t += 1
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                done = terminated | truncated

                ep_rews.append(rew)
                batch_acts.append(action)
                
                batch_log_probs.append(float(log_prob))

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        if self.discrete:
            batch_acts = torch.tensor(batch_acts, dtype=torch.long)
        else:
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs):
        """
            obs: numpy array observation
            returns: action suitable for env.step() (numpy scalar or array), and float log_prob
        """
        obs_t = torch.tensor(obs, dtype=torch.float)
        policy_out = self.actor(obs_t)

        if self.discrete:
            dist = Categorical(logits=policy_out)  
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_np = int(action.item())
            return action_np, float(log_prob.detach().item())
        else:
            mean = policy_out
            std = torch.sqrt(self.cov_var)
            dist = Independent(Normal(mean, std), 1)
            action = dist.rsample()
            log_prob = dist.log_prob(action)
            return action.detach().numpy(), float(log_prob.detach().item())

    def evaluate(self, batch_obs, batch_acts):
        """
            batch_obs: tensor [N, obs_dim]
            batch_acts: tensor [N] (long) for discrete or [N, act_dim] (float) for continuous
            return: V (tensor [N]), log_probs (tensor [N])
        """
        V = self.critic(batch_obs).squeeze()
        policy_out = self.actor(batch_obs)

        if self.discrete:
            dist = Categorical(logits=policy_out)
            if batch_acts.dim() > 1:
                batch_acts = batch_acts.squeeze(dim=-1)
            log_probs = dist.log_prob(batch_acts)
        else:
            mean = policy_out
            std = torch.sqrt(self.cov_var)
            dist = Independent(Normal(mean, std), 1)
            log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def _init_hyperparameters(self, hyperparameters):
        # defaults
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        self.render = True
        self.render_every_i = 10
        self.save_freq = 10
        self.seed = None

        for param, val in hyperparameters.items():
            # Use setattr instead of exec to safely assign arbitrary values
            # (e.g., objects like a wandb Run) as hyperparameters/attributes.
            setattr(self, param, val)

        if self.seed is not None:
            assert(type(self.seed) == int)
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens_num = np.mean(self.logger['batch_lens']) if len(self.logger['batch_lens'])>0 else 0.0
        avg_ep_rews_num = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']]) if len(self.logger['batch_rews'])>0 else 0.0
        avg_actor_loss_num = np.mean([losses.float().mean() for losses in self.logger['actor_losses']]) if len(self.logger['actor_losses'])>0 else 0.0

        self.logger['last_avg_ep_lens'] = float(avg_ep_lens_num)
        self.logger['last_avg_ep_rews'] = float(avg_ep_rews_num)
        self.logger['last_avg_actor_loss'] = float(avg_actor_loss_num)

        avg_ep_lens = str(round(avg_ep_lens_num, 2))
        avg_ep_rews = str(round(avg_ep_rews_num, 2))
        avg_actor_loss = str(round(avg_actor_loss_num, 5))

        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
