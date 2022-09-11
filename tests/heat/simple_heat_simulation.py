""" Heat Relaxation

A horizontal plate at the top is heated and a sphere at the bottom is cooled.
Control the heat source using the sliders at the bottom.
"""
from matplotlib import pyplot as plt
from phi.flow import *
from tqdm import tqdm

N = 128
domain_dict = dict(x=N, extrapolation=extrapolation.PERIODIC, bounds=Box(x=5))
t = CenteredGrid(lambda x: math.sin(x), **domain_dict)
diffusivity = 0.01
dt = 0.05
step_count = 2000
trajectory = [t.vector['x']]
velocities = [t]
for i in tqdm(range(step_count)):
    t = diffuse.explicit(t, dt=dt, diffusivity=diffusivity)
    trajectory.append(t.vector['x'])
    # if i % (step_count // 4) == 0:
        # vis.show(t)
    velocities.append(t)
temp_t = field.stack(trajectory, spatial('time'), Box(time=len(trajectory) * dt))  # time=len(trajectory)* dt
vis.show(temp_t.vector[0], aspect='auto', size=(8, 3))

# get "velocity.values" from each phiflow state with a channel dimensions, i.e. "vector"
vels = [v.values.numpy('x,vector') for v in velocities]  # gives a list of 2D arrays

import pylab

fig = pylab.figure().gca()
fig.plot(np.linspace(-1, 1, len(vels[0].flatten())), vels[0].flatten(), lw=2, color='blue', label="t=0")
fig.plot(np.linspace(-1, 1, len(vels[step_count//3].flatten())), vels[step_count//3].flatten(), lw=2, color='green', label="t=0.3125")
fig.plot(np.linspace(-1, 1, len(vels[step_count*2//3].flatten())), vels[step_count*2//3].flatten(), lw=2, color='cyan', label="t=0.625")
fig.plot(np.linspace(-1, 1, len(vels[step_count-1].flatten())), vels[step_count-1].flatten(), lw=2, color='purple', label="t=1")
pylab.xlabel('x')
pylab.ylabel('t')
pylab.legend()
pylab.show()