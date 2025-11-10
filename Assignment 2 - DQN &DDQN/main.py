from dqn import DQNAgent, DDQNAgent
import torch
import numpy as np
import gymnasium as gym
import wandb
import math

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
    import time
    import math

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        agent = DQNAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
        )
    elif algo.lower() == "ddqn":
        agent = DDQNAgent(
            state_dim=obs_dim,
            action_dim=action_dim,
            lr=lr,
            gamma=gamma,
            target_update_freq=target_update_freq,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Optional: initialize WandB
    if use_wandb:
        wandb.init(project=wandb_project, config={
            "env": env_name,
            "algo": algo,
            "lr": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "replay_size": replay_size,
            "start_epsilon": start_epsilon,
            "end_epsilon": end_epsilon,
            "eps_decay_steps": eps_decay_steps,
        })

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

            # Train if replay buffer has enough samples
            if len(agent.replay_buffer) > max(min_replay_size, batch_size):
                agent.optimize_step(batch_size)

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        # Logging
        if use_wandb:
            wandb.log({
                "episode": episode,
                "epsilon": epsilon,
                "reward": ep_reward,
                "length": ep_length,
            })

        # Evaluation printout
        if episode % eval_every == 0 or episode == episodes:
            avg_reward = np.mean(episode_rewards[-eval_every:])
            avg_length = np.mean(episode_lengths[-eval_every:])
            print(f"[{episode}/{episodes}] "
                  f"Avg Reward (last {eval_every}): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {epsilon:.3f}")

            if use_wandb:
                wandb.log({"avg_reward": avg_reward, "avg_length": avg_length})

    env.close()
    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.1f}s. "
          f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")

    if use_wandb:
        wandb.finish()

    # Return for later analysis
    return agent, {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "training_time": total_time,
    }


def evaluate(agent, env_name: str, episodes: int = 10, max_steps: int = 1000, seed: int = 0):
    """
    Evaluate a trained DQN or DDQN agent.
    Runs the agent in greedy mode (epsilon = 0.0) for given episodes.
    Returns (mean_reward, list_of_rewards)
    """
    import numpy as np
    import gymnasium as gym

    env = gym.make(env_name)
    all_returns = []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        for t in range(max_steps):
            # Greedy action (no exploration)
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

def record_playback(agent, env_name: str, video_dir: str = "./videos", episodes: int = 3, max_steps: int = 1000):
    """
    Records videos of a trained agent interacting with the environment.
    Compatible with both DQNAgent and DDQNAgent.
    """
    import gymnasium as gym
    import os

    # Use gymnasium or fallback to gym
    try:
        from gymnasium.wrappers import RecordVideo
    except ImportError:
        from gym.wrappers import RecordVideo

    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(env_name)
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


if __name__ == "__main__":
    from dqn import DQNAgent, DDQNAgent

    # Train
    agent, stats = train(env_name="CartPole-v1", algo="ddqn", episodes=300, use_wandb=False)

    # Evaluate
    evaluate(agent, env_name="CartPole-v1", episodes=10)

    # Record video
    record_playback(agent, env_name="CartPole-v1", video_dir="./videos_ddqn", episodes=2)
