
import d4rl
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
env = gym.make('maze2d-medium-v1')

# Initialize the SAC model
model = SAC("MlpPolicy", env, tensorboard_log="./sac_maze2d_medium_tensorboard/")

# Train the model
total_timesteps = 1000000
model.learn(total_timesteps=total_timesteps, log_interval=10)

# Save the trained model
model.save("sac_maze2d_medium")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close the environment
env.close()