import numpy as np
from scipy.integrate import odeint

# Disease parameters
disease_params = {
    "beta": 0.02,
    "gamma": 1/7,
    "schools_closed": False
}
seeds=1
N_c=3666
N_a=7334


# Contact Matrix
contact_matrix = [[18, 9], [3, 12]]
contact_matrix_schools_closed = [[0, 5], [2, 8]]

def contacts(i, j, schools_closed):
    if schools_closed:
        return contact_matrix_schools_closed[i][j]
    else:
        return contact_matrix[i][j]


# Transition rates
def foi(i, params, Ns, ac_idx, acs, schools_closed):
    foi_value = 0

    for ac_j in range(acs):
        foi_value += params["beta"] * contacts(ac_idx, ac_j, schools_closed) * i[ac_j] / Ns[ac_j]
        
    return foi_value

def i_r(params):
    return params["gamma"]

def initialise_modelstate(seeds, N_c, N_a):

    # S_c, I_c, R_c, S_a, I_a, R_a
    return np.array([N_c - seeds, seeds, 0, N_a - seeds, seeds, 0])

# ODE Solver

def ode_system(y0, t, parameters):

    params = parameters["disease_params"]
    Ns = parameters["Ns"]

    s_c, i_c, r_c, s_a, i_a, r_a = y0
    ds_c, di_c, dr_c, ds_a, di_a, dr_a = 0, 0, 0, 0, 0, 0

    # Calculating the rates of change for children and adults   
    ds_c += (- foi([i_c, i_a], params, Ns, 0, 2, params["schools_closed"]) * s_c)
    di_c += foi([i_c, i_a], params, Ns, 0, 2, params["schools_closed"]) * s_c - i_r(params) * i_c
    dr_c += i_r(params) * i_c

    ds_a += (- foi([i_c, i_a],  params, Ns, 1, 2, params["schools_closed"]) * s_a)
    di_a += foi([i_c, i_a], params, Ns, 1, 2, params["schools_closed"]) * s_a - i_r(params) * i_a
    dr_a +=  i_r(params) * i_a

    return ds_c, di_c, dr_c, ds_a, di_a, dr_a    

def run_sir_model(model_state, end_t, params, Ns):
    all_parameters = {
        "disease_params": params,
        "Ns": Ns
    }

    # Initial conditions (modelstates, timesteps)   
    y0 = (*model_state,)
    t = np.linspace(0, end_t, end_t)
    
    # Solving the ODE system
    ret = odeint(ode_system, y0, t, args=(all_parameters,))
    s_c, i_c, r_c, s_a, i_a, r_a = ret.T

    return np.array([s_c[-1], i_c[-1], r_c[-1], s_a[-1], i_a[-1], r_a[-1]])
