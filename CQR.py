import d4rl
import gym
import numpy as np
import pickle
import csv
import sys
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from sklearn.model_selection import train_test_split
from huggingface_sb3 import load_from_hub
from sklearn.ensemble import GradientBoostingRegressor
from mujoco_py import MjSimState

dataset = sys.argv[1]

# 'states' and 'reward' predictions from the diffuser 
file = open("data/diffuser_" + dataset + "_rewards", 'rb')
reward_predictions = pickle.load(file)
file.close()

file = open("data/diffuser_" + dataset + "_states", 'rb')
states = pickle.load(file)
file.close()

file = open("data/SAC_"+ dataset +"_rewards", 'rb')
rewards = pickle.load(file)
file.close()

file = open("data/calib_states_index", 'rb')
calib_states_index = pickle.load(file)
file.close()

file = open("data/test_states_index", 'rb')
test_states_index = pickle.load(file)
file.close()

file = open("data/diffuser_train_" + dataset + "_states", 'rb')
X_train = pickle.load(file)
file.close()

file = open("data/diffuser_train_" + dataset + "_rewards", 'rb')
y_train = pickle.load(file)
file.close()

X_cal = [states[index] for index in calib_states_index]
y_cal = [rewards[index] for index in calib_states_index]
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

#fitting the models
lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)
np.random.seed(42)
torch.manual_seed(42)

#calculate the nonconformity scores
def compute_conformity_scores(model, X, y, quantile):
    predictions = model.predict(X)
    if quantile < 0.5:
        return np.maximum(predictions - y, 0)
    else:
        return np.maximum(y - predictions, 0)

#calibrating the data
def calculate_metrics(lower_model, upper_model, X_cal, y_cal, X_test, y_test):

    lower_scores = compute_conformity_scores(lower_model, X_cal, y_cal, lower_quantile)
    upper_scores = compute_conformity_scores(upper_model, X_cal, y_cal, upper_quantile)
    cal_scores = np.maximum(lower_scores, upper_scores)

    n = len(X_cal)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')

    lower_pred = lower_model.predict(X_test) - qhat
    upper_pred = upper_model.predict(X_test) + qhat

    coverage = np.mean((lower_pred <= y_test) & (y_test <= upper_pred))
    avg_width = np.mean(upper_pred - lower_pred)
    return coverage, avg_width

coverage, avg_width = calculate_metrics(lower_model, upper_model, X_cal, y_cal, X_test, y_test)

print(f"CQR Coverage: {coverage:.2f}")
print(f"CQR Average interval width: {avg_width:.4f}")

#data for calibration size impact
def plot_calibration_size_impact(lower_model, upper_model, X_cal, y_cal, X_test, y_test):
    alpha = 0.1
    R = 100
    
    calib_sizes = []
    calib_data_over_R_CQR = {}
    test_size = 200
    for calib_size in range(100, 900, 100):
        coverages_cs = []*R
        interval_widths_cs = []*R
        for r in range(R):
            np.random.shuffle(X_cal)
            np.random.shuffle(y_cal)
            np.random.shuffle(X_test)
            np.random.shuffle(y_test)
            coverage, interval_width = calculate_metrics(lower_model, upper_model, X_cal[:calib_size], y_cal[:calib_size], X_test, y_test)
            coverages_cs.append(coverage)
            interval_widths_cs.append(interval_width)
            calib_sizes.append(calib_size)
        calib_data_over_R_CQR[calib_size] = (coverages_cs, interval_widths_cs)

    filehandler = open("data/calib_size_over_R_CQR_" + dataset, "wb")
    pickle.dump(calib_data_over_R_CQR, filehandler)
    filehandler.close()
    return coverages_cs, interval_widths_cs

coverages_cs, interval_widths_cs = plot_calibration_size_impact(lower_model, upper_model, X_cal, y_cal, X_test, y_test)

#compute the nonconformity scores for upper and lower models
def get_scores(X, Y):
    lower_scores = compute_conformity_scores(lower_model, X, Y, lower_quantile)
    upper_scores = compute_conformity_scores(upper_model, X, Y, upper_quantile)
    return np.maximum(lower_scores, upper_scores)

# Combine calibration and validation data
X = X_cal + X_test
Y = y_cal + y_test
n = len(X)//2  # number of calibration points
n_val = len(X_test)  # number of validation points

scores = get_scores(X, Y)

# calculate the coverage R times and store in list
R = 100  # number of repetitions
coverages = np.zeros((R,))
interval_widths = np.zeros((R,))
for r in range(R):
    np.random.shuffle(scores)  # shuffle
    calib_scores, val_scores = scores[:n], scores[n:]  # split
    qhat = np.quantile(calib_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')  # calibrate
    coverages[r] = (val_scores <= qhat).astype(float).mean()  
    
    # Calculate interval width for validation set
    lower_pred = lower_model.predict(X_test) - qhat
    upper_pred = upper_model.predict(X_test) + qhat
    interval_widths[r] = np.mean(upper_pred - lower_pred)


average_coverage = coverages.mean()
average_interval_width = interval_widths.mean()
print(f"Average coverage: {average_coverage:.4f} ± {np.std(coverages):.4f}")
print(f"Average interval width: {average_interval_width:.4f} ± {np.std(interval_widths):.4f}")

#writing the results to a txt file
with open('resultsCQR.txt', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([dataset])
    writer.writerow([f"Coverage Probability: {coverage:.2f}"])
    writer.writerow([f"Interval Width: {avg_width:.4f}"])
    writer.writerow([f"Average Coverage: {average_coverage:.4f} ± {np.std(coverages):.4f}"])
    writer.writerow([f"Average interval width: {average_interval_width:.4f} ± {np.std(interval_widths):.4f}"])
    writer.writerow([f"Coverage on different calib set sizes: {coverages_cs}"])
    writer.writerow([f"Interval widths on different calib set sizes: {interval_widths_cs}"])
    writer.writerow([])

