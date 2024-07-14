import d4rl
import gym
import numpy as np
import pickle
from stable_baselines3 import SAC
from sklearn.model_selection import train_test_split
from huggingface_sb3 import load_from_hub
from sklearn.ensemble import GradientBoostingRegressor
import torch
from mujoco_py import MjSimState

dataset = 'hopper-medium-replay-v2'

# 'states' and 'rewardpredictions' from the diffuser 
file = open("data/diffuser" + dataset + "_rewards", 'rb')
reward_predictions = pickle.load(file)
file.close()

file = open("data/diffuser_" + dataset + "_states", 'rb')
states = pickle.load(file)
file.close()

file = open("data/SAC_hopper_rewards", 'rb')
rewards = pickle.load(file)
file.close()

file = open("data/calib_states_index", 'rb')
calib_states_index = pickle.load(file)
file.close()

file = open("data/test_states_index", 'rb')
test_states_index = pickle.load(file)
file.close()

train_states_index, calibindex, , _ = train_test_split(
    calib_states_index, calib_states_index, test_size=0.1, random_state=42
)

X_train = [states[index] for index in train_states_index]
y_train = [reward_predictions[index] for index in train_states_index]
X_cal = [states[index] for index in calib_index]
y_cal = [rewards[index] for index in calib_index]
X_test = [states[index] for index in test_states_index]
y_test = [rewards[index] for index in test_states_index]

print(len(X_train), len(X_cal), len(X_test))
# alpha for 90% coverage
alpha = 0.1
lower_quantile = alpha / 2
upper_quantile = 1 - alpha / 2

#Reference: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
lower_model = GradientBoostingRegressor(
    loss='quantile', alpha=lower_quantile, 
    n_estimators=100, learning_rate=0.05, 
    max_depth=5, min_samples_leaf=50, random_state=42
)
upper_model = GradientBoostingRegressor(
    loss='quantile', alpha=upper_quantile, 
    n_estimators=100, learning_rate=0.05, 
    max_depth=5, min_samples_leaf=50, random_state=42
)

lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)

def compute_conformity_scores(model, X, y, quantile):
    predictions = model.predict(X)
    #print(predictions)
    #print("Y:", y)
    if quantile < 0.5:
        return np.maximum(predictions - y, 0)
    else:
        return np.maximum(y - predictions, 0)

lower_scores = compute_conformity_scores(lower_model, X_cal, y_cal, lower_quantile)
upper_scores = compute_conformity_scores(upper_model, X_cal, y_cal, upper_quantile)
cal_scores = np.maximum(lower_scores, upper_scores)

qhat = np.quantile(cal_scores, (1 - alpha))

lower_pred = lower_model.predict(X_test) - qhat
upper_pred = upper_model.predict(X_test) + qhat

coverage = np.mean((lower_pred <= y_test) & (y_test <= upper_pred))
avg_width = np.mean(upper_pred - lower_pred)

print(f"CQR Coverage: {coverage:.2f}")
print(f"CQR Average interval width: {avg_width:.2f}")





def get_scores(X, Y):
    lower_scores = compute_conformity_scores(lower_model, X, Y, lower_quantile)
    upper_scores = compute_conformity_scores(upper_model, X, Y, upper_quantile)
    return np.maximum(lower_scores, upper_scores)

# Combine calibration and validation data
X = X_cal + X_test
Y = y_cal + y_test
n = len(X_cal)  # number of calibration points
n_val = len(X_test)  # number of validation points

scores = get_scores(X, Y)

# calculate the coverage R times and store in list
R = 1000  # number of repetitions
coverages = np.zeros((R,))

for r in range(R):
    np.random.shuffle(scores)  # shuffle
    calib_scores, val_scores = scores[:n], scores[n:]  # split
    qhat = np.quantile(calib_scores, np.ceil((n+1)*(1-alpha)/n)/n, method='higher')  # calibrate
    coverages[r] = (val_scores <= qhat).astype(float).mean()  # see caption

average_coverage = coverages.mean()  # should be close to 1-alpha
print(f"Average coverage: {average_coverage:.4f}")

plt.hist(coverages)  # should be roughly centered at 1-alpha
plt.title("Distribution of Coverages")
plt.xlabel("Coverage")
plt.ylabel("Frequency")
plt.axvline(x=1-alpha, color='r', linestyle='--', label='Target Coverage (1-alpha)')
plt.legend()
plt.show()