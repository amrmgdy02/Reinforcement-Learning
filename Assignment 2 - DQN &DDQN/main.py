from dqn import DQNAgent, DDQNAgent, BaseAgent
import torch
import numpy as np
import gymnasium as gym
import wandb
import math
import optuna
import time
import os
import matplotlib.pyplot as plt
try:
    from gymnasium.wrappers import RecordVideo
except ImportError:
    from gym.wrappers import RecordVideo

# -------------------------------
# Discretize continuous actions
# -------------------------------
class DiscretizedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins=11):
        super().__init__(env)
        self.n_bins = n_bins
        # create discrete actions evenly spaced in original action space
        self.discrete_actions = np.linspace(env.action_space.low[0], env.action_space.high[0], n_bins)
        self.action_space = gym.spaces.Discrete(n_bins)

    def action(self, action_idx):
        return np.array([self.discrete_actions[action_idx]], dtype=np.float32)

# -------------------------------
# Update train and evaluate to use wrapper if needed
# -------------------------------
def make_env(env_name: str, render_mode=None):
    if render_mode:
        env = gym.make(env_name, render_mode=render_mode)
    else:
        env = gym.make(env_name)
    if env_name == "Pendulum-v1":
        env = DiscretizedActionWrapper(env, n_bins=11)
    return env




# -------------------------------
# Train function
# -------------------------------
def train(
    env_name: str = "CartPole-v1",
    algo: str = "ddqn",
    episodes: int = 400,
    max_steps_per_episode: int = 1000,
    hidden_dims: tuple = (128, 128),
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    replay_size: int = 100_000,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.02,
    eps_decay_steps: int = 30000,
    target_update_freq: int = 1000,
    min_replay_size: int = 1000,
    eval_every: int = 50,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "rl_assignment",
    device_str: str = None,
):
    device = torch.device(device_str if device_str else "cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = make_env(env_name)
    obs, _ = env.reset(seed=seed)
    obs_dim = np.array(obs).flatten().shape[0]

    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    else:
        raise ValueError("Only discrete-action environments are supported.")

    # Initialize agent
    if algo.lower() == "dqn":
        agent = DQNAgent(state_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims, lr=lr, gamma=gamma)
    elif algo.lower() == "ddqn":
        agent = DDQNAgent(state_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims, lr=lr, gamma=gamma, target_update_freq=target_update_freq)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Optional WandB
    if use_wandb:
        wandb.init(project=wandb_project, 
                   name=f"train_{algo}_{env_name}_{len(hidden_dims)}_layers_lr{lr}", 
                   config={
            "env": env_name, "algo": algo, "lr": lr, "gamma": gamma,
            "batch_size": batch_size, "replay_size": replay_size,
            "start_epsilon": start_epsilon, "end_epsilon": end_epsilon,
            "eps_decay_steps": eps_decay_steps,
        })

    # Epsilon schedule
    def get_epsilon(step):
        return end_epsilon + (start_epsilon - end_epsilon) * math.exp(-1.0 * step / eps_decay_steps)

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    episode_durations = []
    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0
        ep_length = 0
        ep_start_time = time.time()

        for t in range(max_steps_per_episode):
            epsilon = get_epsilon(total_steps)
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            ep_length += 1
            total_steps += 1

            if len(agent.replay_buffer) > max(min_replay_size, batch_size):
                agent.optimize_step(batch_size)

            if done:
                break
        ep_duration = time.time() - ep_start_time
        episode_durations.append(ep_duration)
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if use_wandb:
            wandb.log({
                "episode": episode, "epsilon": epsilon,
                "reward": ep_reward, "length": ep_length,
                "duration": ep_duration,
            })

        if episode % eval_every == 0 or episode == episodes:
            avg_reward = np.mean(episode_rewards[-eval_every:])
            avg_length = np.mean(episode_lengths[-eval_every:])
            print(f"[{episode}/{episodes}] Avg Reward (last {eval_every}): {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Epsilon: {epsilon:.3f}")
            if use_wandb:
                wandb.log({"avg_reward": avg_reward, "avg_length": avg_length})

    env.close()
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s. Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    # log avg reward over last 100 episodes to wandb
    if use_wandb:
        wandb.log({"final_avg_reward_100": np.mean(episode_rewards[-100:])})

    if use_wandb:
        wandb.finish()

    return agent, {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "training_time": total_time,
    }
    
    
def train_agents(
    env_name: str = "CartPole-v1",
    episodes: int = 400,
    max_steps_per_episode: int = 1000,
    hidden_dims: tuple = (128, 128),
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    replay_size: int = 100_000,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.02,
    eps_decay_steps: int = 30000,
    target_update_freq: int = 1000,
    min_replay_size: int = 1000,
    eval_every: int = 50,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "rl_assignment",
    device_str: str = None,
):
    """Train both DQN and DDQN agents on the same environment in a single wandb run."""

    algorithms = ["dqn", "ddqn"]
    results = {}

    device = torch.device(device_str if device_str else "cuda" if torch.cuda.is_available() else "cpu")

    # Optional WandB (one run for both agents)
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"compare_DQN_DDQN_{env_name}_{len(hidden_dims)}layers_lr{lr}",
            config={
                "env": env_name, "lr": lr, "gamma": gamma,
                "batch_size": batch_size, "replay_size": replay_size,
                "start_epsilon": start_epsilon, "end_epsilon": end_epsilon,
                "eps_decay_steps": eps_decay_steps, "hidden_dims": hidden_dims
            }
        )

    for algo in algorithms:
        # Initialize environment
        env = make_env(env_name)
        obs, _ = env.reset(seed=seed)
        obs_dim = np.array(obs).flatten().shape[0]
        action_dim = env.action_space.n

        # Initialize agent
        if algo == "dqn":
            agent = DQNAgent(state_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims, lr=lr, gamma=gamma)
        else:  # ddqn
            agent = DDQNAgent(state_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims,
                              lr=lr, gamma=gamma, target_update_freq=target_update_freq)

        # Training loop
        total_steps = 0
        episode_rewards, episode_lengths, episode_durations = [], [], []
        start_time = time.time()

        def get_epsilon(step):
            return end_epsilon + (start_epsilon - end_epsilon) * math.exp(-1.0 * step / eps_decay_steps)

        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            ep_reward, ep_length = 0, 0
            ep_start_time = time.time()

            for t in range(max_steps_per_episode):
                epsilon = get_epsilon(total_steps)
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, next_state, float(done))
                state = next_state
                ep_reward += reward
                ep_length += 1
                total_steps += 1

                if len(agent.replay_buffer) > max(min_replay_size, batch_size):
                    agent.optimize_step(batch_size)

                if done:
                    break

            ep_duration = time.time() - ep_start_time
            episode_durations.append(ep_duration)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            # Log per-episode metrics to wandb with algorithm name
            if use_wandb:
                wandb.log({
                    f"{algo}/episode": episode,
                    f"{algo}/epsilon": epsilon,
                    f"{algo}/reward": ep_reward,
                    f"{algo}/length": ep_length,
                    f"{algo}/duration": ep_duration,
                })

            if episode % eval_every == 0 or episode == episodes:
                avg_reward = np.mean(episode_rewards[-eval_every:])
                avg_length = np.mean(episode_lengths[-eval_every:])
                print(f"[{algo.upper()} {episode}/{episodes}] Avg Reward (last {eval_every}): {avg_reward:.2f} | Avg Length: {avg_length:.2f} | Epsilon: {epsilon:.3f}")
                if use_wandb:
                    wandb.log({
                        f"{algo}/avg_reward": avg_reward,
                        f"{algo}/avg_length": avg_length
                    })

        env.close()
        total_time = time.time() - start_time
        print(f"{algo.upper()} training complete in {total_time:.1f}s. Final avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")

        if use_wandb:
            wandb.log({f"{algo}/final_avg_reward_100": np.mean(episode_rewards[-100:])})

        # Save results per agent
        results[algo] = {
            "agent": agent,
            "rewards": episode_rewards,
            "lengths": episode_lengths,
            "durations": episode_durations,
            "training_time": total_time
        }

    if use_wandb:
        wandb.finish()

    return results


# -------------------------------
# Evaluation
# -------------------------------
def evaluate(
    agent: BaseAgent,
    env_name: str,
    episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "rl_assignment",
    algo: str = "ddqn",
    lr: float = 1e-3,
):
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"eval_{algo}_{env_name}_lr{lr}",
            tags=["evaluation", algo, env_name],
            config={
                "env": env_name,
                "algo": algo,
                "lr": lr,
                "episodes": episodes,
            },
        )

    env = make_env(env_name)
    all_returns, episode_lengths, episode_durations = [], [], []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward, ep_length = 0.0, 0
        start_time = time.time()

        for _ in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1
            state = next_state
            if done:
                break

        duration = time.time() - start_time
        all_returns.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_durations.append(duration)

        if use_wandb:
            wandb.log({
                "episode": ep + 1,
                "return": ep_reward,
                "length": ep_length,
                "duration": duration,
            })

    env.close()
    mean_return = np.mean(all_returns)
    mean_length = np.mean(episode_lengths)

    print(f"\n✅ Mean Return: {mean_return:.2f} | Mean Length: {mean_length:.2f}")

    if use_wandb:
        wandb.log({"mean_return": mean_return, "mean_length": mean_length})
        wandb.finish()

    return mean_return, all_returns, episode_lengths, episode_durations


def evaluate_agents(
    agents: dict,
    env_name: str,
    episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    use_wandb: bool = False,
    wandb_project: str = "rl_assignment",
):
    """
    Evaluate multiple agents in the same environment and same WandB run.
    
    agents: dict of {algo_name: agent_instance}, e.g. {"dqn": dqn_agent, "ddqn": ddqn_agent}
    """
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=f"eval_agents_{env_name}",
            tags=["evaluation", env_name],
            config={
                "env": env_name,
                "episodes": episodes,
                "agents": list(agents.keys())
            },
        )

    results = {}

    for algo, agent in agents.items():
        env = make_env(env_name)
        all_returns, episode_lengths, episode_durations = [], [], []

        for ep in range(episodes):
            state, _ = env.reset(seed=seed + ep)
            ep_reward, ep_length = 0.0, 0
            start_time = time.time()

            for _ in range(max_steps):
                action = agent.select_action(state, epsilon=0.0)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
                state = next_state
                if done:
                    break

            duration = time.time() - start_time
            all_returns.append(ep_reward)
            episode_lengths.append(ep_length)
            episode_durations.append(duration)

            if use_wandb:
                wandb.log({
                    f"{algo}/episode": ep + 1,
                    f"{algo}/return": ep_reward,
                    f"{algo}/length": ep_length,
                    f"{algo}/duration": duration,
                })

        env.close()
        mean_return = np.mean(all_returns)
        mean_length = np.mean(episode_lengths)

        print(f"[{algo.upper()}] Mean Return: {mean_return:.2f} | Mean Length: {mean_length:.2f}")

        if use_wandb:
            wandb.log({
                f"{algo}/mean_return": mean_return,
                f"{algo}/mean_length": mean_length
            })

        results[algo] = {
            "mean_return": mean_return,
            "returns": all_returns,
            "lengths": episode_lengths,
            "durations": episode_durations
        }

    if use_wandb:
        wandb.finish()

    return results


# -------------------------------
# Record Playback
# -------------------------------
def record_playback(
    agent: BaseAgent,
    env_name: str,
    video_dir: str = "./videos",
    file_name: str = "run1",
    episodes: int = 3,
    max_steps: int = 1000
):
    full_dir = os.path.join(video_dir, file_name)
    os.makedirs(full_dir, exist_ok=True)

    try:
        env = make_env(env_name, render_mode="rgb_array")
    except TypeError:
        env = make_env(env_name)

    if getattr(env, "render_mode", None) is None:
        supported = env.metadata.get("render_modes", []) if hasattr(env, "metadata") else []
        if "rgb_array" in supported:
            env.close()
            env = make_env(env_name, render_mode="rgb_array")
        else:
            env.close()
            raise ValueError("Environment must support image render mode 'rgb_array'.")

    env = RecordVideo(env, video_folder=full_dir, episode_trigger=lambda ep: True)
    print(f"🎥 Recording {episodes} episode(s) to: {os.path.abspath(full_dir)}")

    for ep in range(episodes):
        state, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
            if done:
                break
        print(f"Recorded Episode {ep + 1}: return = {ep_reward:.2f}")

    env.close()
    print(f"🎬 Saved in: {os.path.abspath(full_dir)}")

################################################################
# # -------------------------------
# # Environment-specific Optuna objective
# # -------------------------------
# def objective(trial, env_name):

#     # Environment-specific ranges
#     if env_name == "CartPole-v1":
#         lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#         gamma = trial.suggest_float("gamma", 0.90, 0.999)
#         batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
#         replay_size = trial.suggest_int("replay_size", 10_000, 50_000, step=10_000)
#         min_replay_size = trial.suggest_int("min_replay_size", 500, 1000, step=500)
#         eps_decay_steps = trial.suggest_int("eps_decay_steps", 5000, 15_000)
#         episodes = trial.suggest_int("episodes", 150, 300, step=50)
#     elif env_name == "Acrobot-v1":
#         lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#         gamma = trial.suggest_float("gamma", 0.98, 0.999)
#         batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
#         replay_size = trial.suggest_int("replay_size", 50_000, 100_000, step=10_000)
#         min_replay_size = trial.suggest_int("min_replay_size", 2000, 5000, step=500)
#         eps_decay_steps = trial.suggest_int("eps_decay_steps", 30_000, 80_000)
#         episodes = trial.suggest_int("episodes", 400, 800, step=50)
#     elif env_name == "MountainCar-v0":
#         lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#         gamma = trial.suggest_float("gamma", 0.98, 0.999)
#         batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
#         replay_size = trial.suggest_int("replay_size", 100_000, 250_000, step=10_000)
#         min_replay_size = trial.suggest_int("min_replay_size", 5000, 10000, step=500)
#         eps_decay_steps = trial.suggest_int("eps_decay_steps", 50_000, 150_000)
#         episodes = trial.suggest_int("episodes", 800, 1500, step=100)
#     elif env_name == "Pendulum-v1":
#         lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
#         gamma = trial.suggest_float("gamma", 0.95, 0.999)
#         batch_size = trial.suggest_int("batch_size", 64, 256, step=64)
#         replay_size = trial.suggest_int("replay_size", 100_000, 300_000, step=50_000)
#         min_replay_size = trial.suggest_int("min_replay_size", 3000, 8000, step=1000)
#         eps_decay_steps = trial.suggest_int("eps_decay_steps", 20_000, 80_000)
#         episodes = trial.suggest_int("episodes", 500, 1000, step=100)
#     else:
#         raise ValueError(f"Unsupported environment: {env_name}")

#     agent, _ = train(
#         env_name=env_name,
#         algo="ddqn",
#         episodes=episodes,
#         lr=lr,
#         gamma=gamma,
#         batch_size=batch_size,
#         replay_size=replay_size,
#         min_replay_size=min_replay_size,
#         eps_decay_steps=eps_decay_steps,
#         use_wandb=False
#     )

#     mean_reward, _ = evaluate(agent, env_name=env_name, episodes=10)
#     return mean_reward

# # -------------------------------
# # Tune and train for one environment
# # -------------------------------
# def tune_and_train(env_name, n_trials=20):
#     print(f"\n--- Tuning hyperparameters for {env_name} ---\n")
#     study = optuna.create_study(direction="maximize")
#     study.optimize(lambda trial: objective(trial, env_name), n_trials=n_trials)

#     best_params = study.best_trial.params
#     print(f"\n Best hyperparameters for {env_name}: {best_params}\n")

#     # Train final agent with best hyperparameters
#     agent, stats = train(
#         env_name=env_name,
#         algo="ddqn",
#         episodes=best_params.get("episodes", 300),
#         lr=best_params["lr"],
#         gamma=best_params["gamma"],
#         batch_size=best_params["batch_size"],
#         replay_size=best_params["replay_size"],
#         min_replay_size=best_params.get("min_replay_size", 1000),
#         eps_decay_steps=best_params["eps_decay_steps"],
#         use_wandb=True
#     )

#     # Evaluate and record
#     evaluate(agent, env_name=env_name, episodes=100)
#     record_playback(agent, env_name=env_name, video_dir=f"./videos_{env_name}", episodes=2)

# # -------------------------------
# # Run tuning for each environment
# # -------------------------------
# if __name__ == "__main__":
#     envs = ["Pendulum-v1"]
#     for env_name in envs:
#         tune_and_train(env_name, n_trials=20)  # increase n_trials for more exhaustive search

#######################################################################



# Main: train with predefined best hyperparameters
# -------------------------------
if __name__ == "__main__":
    
    best_params_per_env = {

    #     "CartPole-v1": {
    #         "lr": 0.0000677,  
    #         "gamma": 0.9921,
    #         "batch_size": 32,
    #         "replay_size": 20_000,
    #         "eps_decay_steps": 14_690,
    #         "episodes": 300,       # more suitable for CartPole
    #         "min_replay_size": 1_000
    #     },

    #     "Acrobot-v1": {
    #         "lr": 0.000575,
    #         "gamma": 0.9959,
    #         "batch_size": 256,
    #         "replay_size": 100_000,
    #         "eps_decay_steps": 57_279,
    #         "episodes": 450,
    #         "min_replay_size": 3_000
    #     },
        

        # "MountainCar-v0": {
        #     "lr": 0.000607,
        #     "gamma": 0.9987,
        #     "batch_size": 192,
        #     "replay_size": 200_000,
        #     "eps_decay_steps": 109_556,
        #     "episodes": 1300,
        #     "min_replay_size": 7_000
        # },

        "Pendulum-v1": {
            "lr": 0.000155,
            "gamma": 0.97814,
            "batch_size": 256,
            "replay_size": 250_000,
            "eps_decay_steps": 22_391,
            "episodes": 800,
            "min_replay_size": 6_000
        }
    }

    for env_name, params in best_params_per_env.items():
        print(f"\n--- Training {env_name} with best parameters ---\n")

        # Train both DQN and DDQN in the same run
        results_train = train_agents(
            env_name=env_name,
            episodes=params["episodes"],
            lr=params["lr"],
            hidden_dims=params.get("hidden_dims", (128, 128)),
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            replay_size=params["replay_size"],
            eps_decay_steps=params["eps_decay_steps"],
            min_replay_size=params["min_replay_size"],
            use_wandb=True
        )

        # Evaluate both agents in the same run
        results_eval = evaluate_agents(
            agents={algo: results_train[algo]["agent"] for algo in results_train},
            env_name=env_name,
            use_wandb=True,
            episodes=100
        )

        # Record a few episodes for each agent
        for algo, data in results_train.items():
            # results_train stores a dict per algorithm with keys: 'agent', 'rewards', etc.
            agent = data["agent"]
            experiment_name = f"{algo}_{env_name}_{len(params.get('hidden_dims', (128, 128)))}_layers_lr{params['lr']}"
            record_playback(
                agent,
                env_name=env_name,
                video_dir=f"./videos_{env_name}",
                file_name=experiment_name,
                episodes=2
            )


