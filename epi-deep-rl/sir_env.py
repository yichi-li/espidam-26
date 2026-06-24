import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sir import initialise_modelstate, run_sir_model

class SIREnv(gym.Env):
    def __init__(self, budget, compartments, seeds, N_c, N_a, params):
        self.budget = budget
        self.seeds = seeds
        self.N_c = N_c
        self.N_a = N_a
        self.N = N_c + N_a
        self.params = params
        self.compartments = compartments

        # Define the action and observation space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = spaces.Box(0, max(N_c, N_a), shape=(len(compartments) + 1,), dtype=np.float32)

    # Return the current model state
    def _get_obs(self):
        #the np.maximum avoid that we get small negative numbers
        state = np.maximum(self.model_state, 0)
        return np.append(state, self.used_budget).astype(np.float32)
    
    # This is not mandatory, but can be useful for debugging
    def _get_info(self):
        return {
            "t": self._t,
        }

    # Reset the environment (modelstate, budget and time)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self.used_budget = 0
        self.model_state = initialise_modelstate(self.seeds, self.N_c, self.N_a)
        return self._get_obs(), self._get_info()


    # Perform a step in the environment given an action
    def step(self, action):
        close_schools = (action == 1)

        # If the budget is zero, we cannot close the schools
        if self.used_budget >= self.budget:
            close_schools = False

        if close_schools:
            self.used_budget += 1
        self.params["schools_closed"] = close_schools
            
        end_t = self._get_info()["t"] + 7
        
        new_model_state = run_sir_model(self.model_state, 7, self.params, [self.N_c, self.N_a])

        _new_s = (new_model_state[self.compartments.index("S_c")]+new_model_state[self.compartments.index("S_a")]) 
        _old_s = (self.model_state[self.compartments.index("S_c")]+self.model_state[self.compartments.index("S_a")])

        self.model_state = new_model_state
        self._t = end_t

        # The reward is the new number of infected individuals, which is calculated as the difference between the old and new number of susceptible individuals
        # Because it is a optimization problem, the reward is negative
        reward = -(_old_s-_new_s)
        terminated = self._t >= 180

        return self._get_obs(), reward, terminated, False, self._get_info()
    
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