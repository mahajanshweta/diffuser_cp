
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Create the environment
env = gym.make('maze2d-medium-v1')

# Create the SAC model
model = SAC('MlpPolicy', env, verbose=1)

# Set up a callback to save the model every 10,000 steps
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='logs/',
                                         name_prefix='sac_maze2d_model')

# Set up an evaluation callback
eval_callback = EvalCallback(env, best_model_save_path='logs/best_model',
                             log_path='logs/results', eval_freq=5000,
                             deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=int(1e6), callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save("sac_maze2d_large")
