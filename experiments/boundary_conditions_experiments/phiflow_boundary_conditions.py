import numpy as np
from phi.flow import *

from src.util import plotGrid

values = math.random_uniform(spatial(x=50))

# bounds = Box[0:1, 0:1]
bounds = Box[0:1]
N = 128

np.random.seed(43)
def simpleUniformRandom(x):
    # return tensor(np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 0.5, self.N), x.shape[0])
    return tensor(np.random.uniform(0, 0.5, N), x.shape[0])


final = []

initial = CenteredGrid(Noise(), x=N, bounds=bounds)
final.append(initial.data.native("vector,x")[0])

initial = CenteredGrid(Noise(), x=N, bounds=bounds, extrapolation=extrapolation.ZERO)
dirichlet = CenteredGrid(initial, x=N, bounds=Box[-2:4]).data.native("vector,x")[0]
final.append(dirichlet)

initial = CenteredGrid(Noise(), x=N, bounds=bounds, extrapolation=extrapolation.BOUNDARY)
neumann = CenteredGrid(initial, x=N, bounds=Box[-2:4]).data.native("vector,x")[0]
final.append(neumann)

initial = CenteredGrid(Noise(), x=N, bounds=bounds, extrapolation=extrapolation.PERIODIC)
periodic = CenteredGrid(initial, x=N, bounds=Box[-2:4]).data.native("vector,x")[0]
final.append(periodic)

prefix = "results/"
labels = ["exper_bc_initial", "exper_bc_dirichlet", "exper_bc_neumann", "exper_bc_periodic"]
for idx, state in enumerate(final):
    plotGrid(listU=[state], saveFig=prefix+labels[idx], ylim_min=-2.0, ylim_max=2.0, domain=N, dx=1, render=True)
