import numpy as np
import phi
from phi.field import CenteredGrid
from phi.geom import Box
from phi.math import extrapolation, tensor
from phi.physics._effect import FieldEffect
from scipy.optimize import minimize
from tqdm import tqdm

from src.env.HeatPhysicsGym import HeatPhysicsGym

ph = 2
dx = 0.25
domain = 4
step_count = 100
domain_dict = dict(x=int(domain / dx), bounds=Box[0:1],
                   extrapolation=extrapolation.BOUNDARY)
env = HeatPhysicsGym(domain=domain, dx=dx, domain_dict=domain_dict,
                     dt=0.1, step_count=step_count,
                     diffusivity=0.1, dxdt=10)
curr_state = env.init_state.data.native("vector,x")[0]
ref_states_np = np.array([env.reference_state_np.flatten() for _ in range(ph)])

shape_nt_state = env.init_state.data.native("x,vector").shape[0]
shape_phi_state = env.init_state.shape


def J(actions: np.ndarray, curr_state: np.ndarray, ref_states: np.ndarray):
    state = CenteredGrid(phi.math.tensor(curr_state, shape_phi_state), **env.domain_dict)
    states = []

    for i in tqdm(range(ph)):
        action = CenteredGrid(
            tensor(env.action_transform(actions[i]).reshape(shape_nt_state), shape_phi_state),
            **env.domain_dict)
        state = env.step_physics(in_state=state, effects=(action,))
        states.append(state.data.native("vector,x")[0])

    dy = np.array(states) - ref_states
    dyQ = 0

    for ii in range(dy.shape[1]):
        dyQ += np.power(dy[:, ii], 2)

    # return env.dt * np.sum(dyQ)
    return np.sum(dyQ)


init_control = np.array([-1.0 for _ in range(ph)])
res = minimize(lambda cont: J(cont, curr_state, ref_states_np),
               init_control, method='SLSQP')  # Nelder-Mead

print("")
