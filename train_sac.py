
import d4rl
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = sys.argv[1]
# Create the environment
env = gym.make(dataset)

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1, device=device)

# Train the model
total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps, log_interval=10)

# Save the trained model
model.save("data/sac_"+ dataset)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()