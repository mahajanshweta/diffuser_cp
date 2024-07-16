import d4rl
import numpy as np
import gym
import csv
from stable_baselines3 import SAC
import torch
from tqdm import tqdm
from huggingface_sb3 import load_from_hub
import pickle
import collections
from mujoco_py import MjSimState
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta


def load_sac_model(repo_id, filename):
    model_path = load_from_hub(repo_id, filename)
    return SAC.load(model_path)

def load_data(dataset):
    with open(f"data50/diffuser_{dataset}_rewards", 'rb') as file:
        reward_predictions = pickle.load(file)
    with open(f"data50/diffuser_{dataset}_states", 'rb') as file:
        states = pickle.load(file)
    return reward_predictions, states

def generate_sac_rewards(env, sac_model, states, num_episodes, num_steps):
    rewards = []
    for i in range(num_episodes):
        obs = env.reset(seed=i)
        udd_state = env.sim.get_state().udd_state
        qpos = np.zeros(env.sim.model.nq)
        qvel = np.zeros(env.sim.model.nv)
        qpos[1:] = states[i][:8] 
        qvel[:] = states[i][8:]
        new_state = MjSimState(env.sim.data.time, qpos, qvel, env.sim.data.act, udd_state)
        env.sim.set_state(new_state)
        env.sim.forward()
        obs = states[i]
        total_reward = 0
        for _ in range(num_steps):
            action, _ = sac_model.predict(obs, deterministic=True)
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            total_reward += reward
        rewards.append(total_reward)
    return rewards

def save_rewards(rewards, dataset):
    with open(f"data50/SAC_{dataset}_rewards", "wb") as filehandler:
        pickle.dump(rewards, filehandler)

def get_sac_rewards(dataset):
    with open(f"data50/SAC_{dataset}_rewards", 'rb') as file:
        rewards = pickle.load(file)
    return rewards     

def split_data(states, rewards, reward_predictions):
    index = np.arange(1000)
    test_states_index, calib_states_index, _, _ = train_test_split(
        index, index, test_size=0.8, random_state=40
    )
    
    test_states = [states[i] for i in test_states_index]
    test_rewards = [rewards[i] for i in test_states_index]
    calib_states = [states[i] for i in calib_states_index]
    calib_rewards = [rewards[i] for i in calib_states_index]
    
    reward_predictions_calib = [reward_predictions[i] for i in calib_states_index]
    reward_predictions_test = [reward_predictions[i] for i in test_states_index]
    
    return (test_states, test_rewards, calib_states, calib_rewards, 
            reward_predictions_calib, reward_predictions_test, 
            calib_states_index, test_states_index)

def save_split_indices(calib_states_index, test_states_index):
    with open("data50/calib_states_index", "wb") as filehandler:
        pickle.dump(calib_states_index, filehandler)
    with open("data50/test_states_index", "wb") as filehandler:
        pickle.dump(test_states_index, filehandler)


def calculate_metrics(calib_rewards, reward_predictions_calib, test_rewards, reward_predictions_test, alpha=0.1):
    residuals = np.abs(np.array(calib_rewards) - np.array(reward_predictions_calib))
    n = len(calib_rewards)
    qhat = np.quantile(residuals, np.ceil((n+1)*(1-alpha))/n, method='higher')
    
    lower_bounds = reward_predictions_test - qhat
    upper_bounds = reward_predictions_test + qhat
    
    coverage = np.mean((test_rewards >= lower_bounds) & (test_rewards <= upper_bounds))
    average_interval_width = np.mean(upper_bounds - lower_bounds)
    
    return coverage, average_interval_width

def get_scores(reward_predictions, rewards):
    return np.abs(np.array(rewards) - np.array(reward_predictions))

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    
    repo_id = "sb3/sac-" + sys.argv[2] + "-v3"
    filename = "sac-" + sys.argv[2] + "-v3.zip"
    dataset = sys.argv[1]
    num_episodes = 1000
    num_steps = 50
    
    sac_model = load_sac_model(repo_id, filename)
    
    env = gym.make(dataset)
    reward_predictions, states = load_data(dataset)
    print(len(reward_predictions), len(states))
    
    rewards = get_sac_rewards(dataset)
    save_rewards(rewards, dataset)
    
    print(f"Average Reward (Diffuser): {np.mean(reward_predictions):.2f} ± {np.std(reward_predictions):.2f}")
    print(f"Average Reward (SAC): {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    
    (test_states, test_rewards, calib_states, calib_rewards, 
     reward_predictions_calib, reward_predictions_test, 
     calib_states_index, test_states_index) = split_data(states, rewards, reward_predictions)
    
    save_split_indices(calib_states_index, test_states_index)
    
    coverage, average_interval_width = calculate_metrics(calib_rewards, reward_predictions_calib, 
                                                         test_rewards, reward_predictions_test)
    
    print(f"Coverage Probability: {coverage:.2f}")
    print(f"Interval Width: {average_interval_width:.4f}")
   
    coverage_calib_set, width_calib_set = plot_calibration_size_impact(calib_rewards, reward_predictions_calib,test_rewards,reward_predictions_test, dataset)

    scores = get_scores(reward_predictions, rewards)
    R=100
    alpha=0.1
    n = len(scores) // 2  # Assuming half for calibration, half for validation
    
    # Calculate the coverage R times and store in list
    coverages = np.zeros((R,))
    interval_widths = np.zeros((R,))
    for r in range(R):
        # Shuffle both scores and rewards (and predictions) in the same way
        shuffle_indices = np.random.permutation(len(scores))
        shuffled_scores = scores[shuffle_indices]
        shuffled_rewards = np.array(rewards)[shuffle_indices]
        shuffled_predictions = np.array(reward_predictions)[shuffle_indices]
        
        calib_scores, val_scores = shuffled_scores[:n], shuffled_scores[n:]
        calib_rewards, val_rewards = shuffled_rewards[:n], shuffled_rewards[n:]
        calib_predictions, val_predictions = shuffled_predictions[:n], shuffled_predictions[n:]
        
        qhat = np.quantile(calib_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')  # calibrate
        
        lower_bounds = val_predictions - qhat
        upper_bounds = val_predictions + qhat
        
        coverages[r] = np.mean((val_rewards >= lower_bounds) & (val_rewards <= upper_bounds))
        interval_widths[r] = np.mean(upper_bounds - lower_bounds)
    
    average_coverage = coverages.mean()  # should be close to 1-alpha
    average_interval_width = interval_widths.mean()
    
    print(f"Average Coverage: {average_coverage:.4f} ± {np.std(coverages):.2f}")
    print(f"Average interval width: {average_interval_width:.4f} ± {np.std(interval_widths):.2f}")


    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([dataset])
        writer.writerow([f"Average Reward (Diffuser): {np.mean(reward_predictions):.2f} ± {np.std(reward_predictions):.2f}"])
        writer.writerow([f"Average Reward (SAC): {np.mean(rewards):.2f} ± {np.std(rewards):.2f}"])
        writer.writerow([f"Coverage Probability: {coverage:.2f}"])
        writer.writerow([f"Interval Width: {average_interval_width:.4f}"])
        writer.writerow([f"Average Coverage: {average_coverage:.4f} ± {np.std(coverages):.2f}"])
        writer.writerow([f"Average interval width: {average_interval_width:.4f} ± {np.std(interval_widths):.2f}"])
        writer.writerow([f"Coverage on different calib set sizes: {coverage_calib_set}"])
        writer.writerow([f"Interval widths on different calib set sizes: {width_calib_set}"])
        writer.writerow([])
    
    '''plt.hist(coverages)
    plt.title("Distribution of Coverage Probabilities")
    plt.xlabel("Coverage")
    plt.ylabel("Frequency")
    plt.axvline(1-alpha, color='r', linestyle='dashed', linewidth=2, label=f'1-alpha ({1-alpha:.2f})')
    plt.legend()
    plt.show()'''

    

   

def plot_calibration_size_impact(calib_rewards, reward_predictions_calib, test_rewards, reward_predictions_test, dataset):
    alpha =0.1
    coverages = []
    interval_widths = []
    calib_sizes = []
    test_size = 200
    for calib_size in range(100, 900, 100):
        coverage, interval_width = calculate_metrics(calib_rewards[:calib_size], reward_predictions_calib[:calib_size], 
                                                         test_rewards[:test_size], reward_predictions_test[:test_size])
        coverages.append(coverage)
        interval_widths.append(interval_width)
        calib_sizes.append(calib_size)
    
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(calib_sizes, coverages, marker='o')
    ax1.set_xlabel('Calibration Set Size')
    ax1.set_ylabel('Coverage')
    ax1.set_title('Coverage vs Calibration Set Size')
    ax1.axhline(y=1-alpha, color='r', linestyle='--', label=f'1-α ({1-alpha})')
    ax1.legend()
    
    ax2.plot(calib_sizes, interval_widths, marker='o')
    ax2.set_xlabel('Calibration Set Size')
    ax2.set_ylabel('Average Interval Width')
    ax2.set_title('Interval Width vs Calibration Set Size')
    
    plt.tight_layout()
    plt.savefig("plots/cp" + dataset + ".png")
    return coverages, interval_widths

if __name__ == "__main__":
    main()