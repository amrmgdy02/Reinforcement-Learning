from dqn import DQNAgent, DDQNAgent, BaseAgent
import torch
import numpy as np
import gymnasium as gym
import wandb
import math
import optuna
import time
import os

# -------------------------------
# Train function
# -------------------------------
def train(
    env_name: str = "CartPole-v1",
    algo: str = "ddqn",
    episodes: int = 400,
    max_steps_per_episode: int = 1000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    replay_size: int = 100_000,
    start_epsilon: float = 1.0,
    end_epsilon: float = 0.02,
    eps_decay_steps: int = 30000,
    target_update_freq: int = 1000,
    min_replay_size: int = 1000,
    eval_every: int = 25,
    eval_episodes: int = 5,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "dqn_project",
    device_str: str = None,
):
    device = torch.device(device_str if device_str else "cuda" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = gym.make(env_name)
    obs, _ = env.reset(seed=seed)
    obs_dim = np.array(obs).flatten().shape[0]

    if hasattr(env.action_space, "n"):
        action_dim = env.action_space.n
    else:
        raise ValueError("Only discrete-action environments are supported.")

    # Initialize agent
    if algo.lower() == "dqn":
        agent = DQNAgent(state_dim=obs_dim, action_dim=action_dim, lr=lr, gamma=gamma)
    elif algo.lower() == "ddqn":
        agent = DDQNAgent(state_dim=obs_dim, action_dim=action_dim, lr=lr, gamma=gamma, target_update_freq=target_update_freq)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Optional WandB
    if use_wandb:
        wandb.init(project=wandb_project, config={
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
    start_time = time.time()

    for episode in range(1, episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        ep_reward = 0
        ep_length = 0

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

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if use_wandb:
            wandb.log({
                "episode": episode, "epsilon": epsilon,
                "reward": ep_reward, "length": ep_length,
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

    if use_wandb:
        wandb.finish()

    return agent, {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "training_time": total_time,
    }

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(agent: BaseAgent, env_name: str, episodes: int = 10, max_steps: int = 1000, seed: int = 0):
    env = gym.make(env_name)
    all_returns = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            state = next_state
            ep_reward += reward
            if done:
                break
        all_returns.append(ep_reward)
        print(f"Episode {ep + 1}: return = {ep_reward:.2f}")

    env.close()
    mean_return = float(np.mean(all_returns))
    print(f"\n✅ Average return over {episodes} episodes: {mean_return:.2f}")
    return mean_return, all_returns

# -------------------------------
# Record Playback
# -------------------------------
def record_playback(agent: BaseAgent, env_name: str, video_dir: str = "./videos", episodes: int = 3, max_steps: int = 1000):
    try:
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        from gym.wrappers import RecordVideo

    os.makedirs(video_dir, exist_ok=True)
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except TypeError:
        env = gym.make(env_name)

    if getattr(env, "render_mode", None) is None:
        supported = env.metadata.get("render_modes", []) if hasattr(env, "metadata") else []
        if "rgb_array" in supported:
            env.close()
            env = gym.make(env_name, render_mode="rgb_array")
        else:
            env.close()
            raise ValueError("Environment must support image render mode 'rgb_array'.")

    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)
    print(f"🎥 Recording {episodes} episode(s) to: {os.path.abspath(video_dir)}")

    for ep in range(episodes):
        state, _ = env.reset(seed=1000 + ep)
        ep_reward = 0.0
        for t in range(max_steps):
            action = agent.select_action(state, epsilon=0.0)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            state = next_state
            ep_reward += reward
            if done:
                break
        print(f"Recorded Episode {ep + 1}: return = {ep_reward:.2f}")

    env.close()
    print(f"🎬 Videos saved in: {os.path.abspath(video_dir)}")

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
#     envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
#     for env_name in envs:
#         tune_and_train(env_name, n_trials=20)  # increase n_trials for more exhaustive search

#######################################################################



# Main: train with predefined best hyperparameters
# -------------------------------
if __name__ == "__main__":
    # Best parameters obtained from previous Optuna runs
    best_params_per_env = {

        "CartPole-v1": {
            "lr": 0.0000677,  
            "gamma": 0.9921,
            "batch_size": 32,
            "replay_size": 20_000,
            "eps_decay_steps": 14_690,
            "episodes": 300,       # more suitable for CartPole
            "min_replay_size": 1_000
        },

        "Acrobot-v1": {
            "lr": 0.000575,
            "gamma": 0.9959,
            "batch_size": 256,
            "replay_size": 100_000,
            "eps_decay_steps": 57_279,
            "episodes": 450,
            "min_replay_size": 3_000
        },
        

        "MountainCar-v0": {
            "lr": 0.000607,
            "gamma": 0.9987,
            "batch_size": 192,
            "replay_size": 200_000,
            "eps_decay_steps": 109_556,
            "episodes": 1300,
            "min_replay_size": 7_000
        },
    }

    for env_name, params in best_params_per_env.items():
        print(f"\n--- Training {env_name} with best parameters ---\n")
        
        agent, stats = train(
            env_name=env_name,
            algo="ddqn",
            episodes=params["episodes"],  
            lr=params["lr"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            replay_size=params["replay_size"],
            eps_decay_steps=params["eps_decay_steps"],
            min_replay_size=params["min_replay_size"],
            use_wandb=True
        )

        # Evaluate on 100 episodes
        evaluate(agent, env_name=env_name, episodes=100)

        # Record a few episodes
        record_playback(agent, env_name=env_name, video_dir=f"./videos_{env_name}", episodes=2)
