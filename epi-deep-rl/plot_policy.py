import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import os

def print_sequence(env: gym.Env, 
                              model: PPO, budget):
    action_sequence = []
    observation, _ = env.reset()
    
    step = 0
    done = False
    used_budget = 0
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        close_schools = (action == 1)
        if used_budget >= budget:
            close_schools = False

        if close_schools:
            used_budget += 1
        action_sequence.append(close_schools)
        observation, _, terminated, _, _ = env.step(action)
        done = terminated
        step += 1
    
    print("Action sequence: ", action_sequence)
    return action_sequence

def plot_policy_with_trajectories(env, model, budget, ppo_seed):
    observation, _ = env.reset()
    actions = []
    states = []
    timesteps = []

    done = False
    step = 0
    used_budget = 0

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        close_schools = (action == 1)
        if used_budget >= budget:
            close_schools = False

        if close_schools:
            used_budget += 1
        actions.append(close_schools)
        states.append((env.model_state[1], env.model_state[4]))
        timesteps.append(step*7)
        observation, _, terminated, _, _ = env.step(action)
        done = terminated
        step += 1

    plt.figure(figsize=(15, 10))
    c_data = [state[0] for state in states]
    a_data = [state[1] for state in states]
    plt.plot(timesteps, c_data, label="Inf. Children", color='red')
    plt.plot(timesteps, a_data, label="Inf. Adults", color='blue')
    plt.subplots_adjust(bottom=0.25)
    xlabels = [f"t: {t}, action: {"close schools" if action == 1 else "open schools"}" for t, action in zip(timesteps, actions)]
    plt.xticks(timesteps, xlabels, rotation=90)
    
    plt.xlabel("Time (days)", fontweight='bold')
    plt.ylabel("Number of Individuals", fontweight='bold')
    plt.title(f"PPO: Actions Taken - Budget {budget}", fontweight='bold')
    plt.legend()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ppo_results_budget_{}_seed_{}.png".format(budget, ppo_seed), dpi=300)
