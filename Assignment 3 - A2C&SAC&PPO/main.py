import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
# Import the agent from the new file
from SAC import SACAgent, ReplayBuffer, Transition 

# -----------------------------
# Hyperparameters (Revised)
# -----------------------------
ENV_ID = "Pendulum-v1"
EPISODES = 400
BATCH_SIZE = 256
REPLAY_SIZE = 300000
GAMMA = 0.99
TAU = 0.005
LR = 3e-4          # Increased from 1e-4
MAX_STEPS = 200
TEST_EPISODES = 10
ALPHA = 0.05         


def train():
    env = gym.make(ENV_ID)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # This is correct: max_action will be 2.0 for Pendulum-v1
    max_action = float(env.action_space.high[0]) 

    sac = SACAgent(obs_dim, act_dim, max_action, lr=LR, gamma=GAMMA, tau=TAU, alpha=ALPHA)
    buffer = ReplayBuffer(REPLAY_SIZE)

    log_interval = 10
    
    for ep in range(EPISODES):
        state, _ = env.reset()
        ep_reward = 0
        critic_loss_sum = 0
        actor_loss_sum = 0
        update_count = 0
        
        for step in range(MAX_STEPS):
            action = sac.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(Transition(state, action, reward, next_state, done))
            state = next_state
            ep_reward += reward

            if len(buffer) > BATCH_SIZE:
                c_loss, a_loss = sac.train(buffer, BATCH_SIZE)
                critic_loss_sum += c_loss
                actor_loss_sum += a_loss
                update_count += 1

            if done:
                break
        
        avg_critic_loss = critic_loss_sum / update_count if update_count > 0 else 0
        avg_actor_loss = actor_loss_sum / update_count if update_count > 0 else 0

        print(f"Episode {ep+1:03d}: Reward = {ep_reward:.2f} | Avg Critic Loss = {avg_critic_loss:.4f} | Avg Actor Loss = {avg_actor_loss:.4f}")

    sac.save("sac_pendulum")
    env.close()


def test():
    print("\nTesting SAC...")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    # Note: Gymnasium will handle the video folder warning.
    env = RecordVideo(env, video_folder="videos_pendulum", name_prefix="sac_test", episode_trigger=lambda x: x < 10)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Reinitialize agent with default parameters (only architecture matters for loading)
    sac = SACAgent(obs_dim, act_dim, max_action) 
    sac.load("sac_pendulum")

    test_rewards = []
    
    for ep in range(TEST_EPISODES):
        # Use deterministic action for testing
        state, _ = env.reset(seed=ep) 
        total_reward = 0
        for step in range(MAX_STEPS):
            action = sac.select_action(state, deterministic=True) 
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        
        test_rewards.append(total_reward)
        print(f"Test Episode {ep+1}: Reward = {total_reward:.2f}")

    avg_test_reward = np.mean(test_rewards)
    print(f"\nAverage Test Reward over {TEST_EPISODES} episodes: {avg_test_reward:.2f}")

    env.close()


if __name__ == "__main__":
    print("Training SAC on Pendulum-v1...")
    #train()
    test()