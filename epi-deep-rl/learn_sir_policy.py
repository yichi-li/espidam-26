import argparse
from sir_env import make_sir_env
import numpy as np
import sir

from gymnasium.wrappers import NormalizeObservation, NormalizeReward

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from plot_policy import print_sequence, plot_policy_with_trajectories

from validate_gym import validate_gym

# Budget parameter
budget = 2

# Make the gym env to learn the policy
env = make_sir_env(budget, seeds=sir.seeds, N_c=sir.N_c, N_a=sir.N_a, gamma=sir.disease_params["gamma"], beta=sir.disease_params["beta"])

# Validate your environment
validate_gym(env, seeds=sir.seeds, N_c=sir.N_c, N_a=sir.N_a, params=sir.disease_params, budget=budget)

# Wrap the environment to normalize observations and rewards
env = NormalizeObservation(env)
env = NormalizeReward(env)

# Setup the PPO model with a seed and tensorboard logging to monitor the training
ppo_seed = np.random.randint(0, 1000000)
ppo_model = PPO(policy="MlpPolicy", 
                env=env, 
                learning_rate=2e-4, 
                verbose=1,
                seed=ppo_seed, 
                tensorboard_log="tensorboard_log")

# Train the model
ppo_model.learn(total_timesteps=35000, tb_log_name=f"ppo_budget_{budget}/seed_{ppo_seed}")

# Show the results
mean_reward, std_reward = evaluate_policy(model=ppo_model, 
                                            env=env, 
                                            deterministic=True)
print(f'PPO with budget={budget}')
print(f"Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")

# Print the learned policy
print_sequence(env, ppo_model, budget)

# Plot the policy
plot_policy_with_trajectories(env, ppo_model, budget, ppo_seed)

#Close the environment
env.close()