import random
from sir import run_sir_model
import sir
import matplotlib.pyplot as plt
import gymnasium as gym
from sir import initialise_modelstate



# Function to create the gym environment
def make_sir_env(budget, seeds, N_c, N_a, gamma, beta):
    # Register the environment
    gym.envs.registration.register(
        id="SIREnv-v0",
        entry_point="sir_env:SIREnv",
        max_episode_steps=300
    )

    compartments = ["S_c", "I_c", "R_c", "S_a", "I_a", "R_a"]
    
    params = {
        "beta": beta, #transmission rate
        "gamma": gamma, #recovery rate
        "schools_closed": False 
    }

    #Make the environment
    env = gym.make("SIREnv-v0",
                    compartments=compartments,
                    seeds=seeds,
                    budget=budget,
                    N_c=N_c,
                    N_a=N_a,
                    params=params)
    
    return env

# Function to validate the gym environment
def validate_gym(env, seeds, N_c, N_a, params, budget=1000):
    env.reset()

    actions = [random.choice([True, False]) for _ in range(50)]

    initial_modelstate = initialise_modelstate(seeds, N_c, N_a)

    env_states = [(initial_modelstate[1], initial_modelstate[4])]

    ode_states = [(initial_modelstate[1], initial_modelstate[4])]
    ode_current_state = initial_modelstate.copy()

    timesteps = [0]

    done = False
    step = 0  
    used_budget = 0
    while not done:
        action = actions[step]
        close_schools = (action == 1)
        if used_budget >= budget:
            close_schools = False

        if close_schools:
            used_budget += 1
        params["schools_closed"] = close_schools
        observation, _, terminated, _, _ = env.step(action)
        ode_current_state = run_sir_model(ode_current_state, 7, params, [N_c, N_a])
        env_states.append((observation[1], observation[4]))
        ode_states.append((ode_current_state[1], ode_current_state[4]))
        done = terminated
        step += 1
        timesteps.append(step*7)

    plt.figure(figsize=(15, 10))

    c_env_data = [state[0] for state in env_states]
    a_env_data = [state[1] for state in env_states]
    c_ode_data = [state[0] for state in ode_states]
    a_ode_data = [state[1] for state in ode_states]

    plt.plot(timesteps, c_env_data, label="I_c - GymEnv", color='red')
    plt.plot(timesteps, a_env_data, label="I_a - GymEnv", color='orange')
    plt.plot(timesteps, c_ode_data, label="I_c - ODE", color='blue', linestyle='dashed')
    plt.plot(timesteps, a_ode_data, label="I_a - ODE", color='turquoise', linestyle='dashed')

    plt.subplots_adjust(bottom=0.25)
    xlabels = [f"t: {t}, action: {"close schools" if action == 1 else "open schools"}" for t, action in zip(timesteps, actions)]
    plt.xticks(timesteps, xlabels, rotation=90)
    
    plt.xlabel("Time (days)", fontweight='bold')
    plt.ylabel("Number of Individuals", fontweight='bold')
    plt.title(f"Gym Validation - ODE vs Gym", fontweight='bold')
    plt.legend()

    plt.show()


# Run the validation
if __name__ == "__main__":
    budget = 2
    env = make_sir_env(budget, sir.seeds, sir.N_c, sir.N_a, sir.disease_params["gamma"], sir.disease_params["beta"])
    validate_gym(env, sir.seeds, sir.N_c, sir.N_a, sir.disease_params, budget)
    env.close()
