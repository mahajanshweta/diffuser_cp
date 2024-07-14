
import d4rl
import numpy as np
import gym
from stable_baselines3 import SAC
from value_guided_sampling import ValueGuidedRLPipeline
import torch
from tqdm import tqdm
from huggingface_sb3 import load_from_hub
import pickle
import collections
from mujoco_py import MjSimState

repo_id = "sb3/sac-Hopper-v3"
filename = "sac-Hopper-v3.zip"

#Loading a pretrained SAC model
model_path = load_from_hub(repo_id, filename)
model = SAC.load(model_path)
print(model)

dataset = "hopper-medium-v2"  
env = gym.make(dataset)

sac_model = model

file = open("data/diffuser_" + dataset + "_rewards", 'rb')
reward_predictions = pickle.load(file)
file.close()

file = open("data/diffuser_" + dataset + "_states", 'rb')
states = pickle.load(file)
file.close()

print(len(reward_predictions), len(states))

rewards = []
num_of_steps = 50
np.random.seed(42)
torch.manual_seed(42)
rewards = []

#Building the ground truth using the SAC model, running for 1000 episodes and 50 steps in each episode
for i in range(len(states)):
    obs = env.reset(seed=i)
    udd_state = env.sim.get_state().udd_state
    qpos = np.zeros(env.sim.model.nq)
    qvel = np.zeros(env.sim.model.nv)
    qpos[1:] = states[i][:5] 
    qvel[:] = states[i][5:]
    new_state = MjSimState(env.sim.data.time, qpos, qvel, env.sim.data.act, udd_state)

    env.sim.set_state(new_state)
    env.sim.forward()
    obs = states[i]
    total_reward = 0
    for j in range(num_of_steps):

        action, _ = sac_model.predict(obs, deterministic=True)
        nextobs, reward, done,  = env.step(action)
        obs = next_obs
        total_reward += reward
    rewards.append(total_reward)

print(f"Average Reward (Diffuser): {np.mean(reward_predictions):.2f} ± {np.std(reward_predictions):.2f}")
print(f"Average Reward (SAC): {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

index = np.arange(start=0, stop=1000, step=1)

from sklearn.model_selection import train_test_split
import operator
test_states_index, calib_statesindex, , _ = train_test_split(
    index, index, test_size=0.8, random_state=40
)

test_states = [states[index] for index in test_states_index]
test_rewards = [rewards[index] for index in test_states_index]
calib_states = [states[index] for index in calib_states_index]
calib_rewards = [rewards[index] for index in calib_states_index]

filehandler = open("data/calib_states_index","wb")
pickle.dump(calib_states_index,filehandler)
filehandler.close()

filehandler = open("data/test_states_index","wb")
pickle.dump(test_states_index,filehandler)
filehandler.close()

#Splitting the rewards into calibration and test for diffuser
reward_predictions_calib = [reward_predictions[index] for index in calib_states_index]
reward_predictions_test = [reward_predictions[index] for index in test_states_index]

#calculating the residual and qhat which is the 1-alpha quantile of residual

residuals = np.abs(np.array(calib_rewards) - np.array(reward_predictions_calib))

alpha = 0.1
n = len(calib_rewards)
qhat = np.quantile(residuals, np.ceil((n+1)*(1-alpha))/n, method='higher')

lower_bounds = reward_predictions_test - qhat
upper_bounds = reward_predictions_test + qhat

coverage = np.mean((test_rewards >= lower_bounds) & (test_rewards <= upper_bounds))
average_interval_width = np.mean(upper_bounds - lower_bounds)

print(f"Coverage Probability: {coverage:.2f}")
print(f"Interval Width: {average_interval_width:.4f}")