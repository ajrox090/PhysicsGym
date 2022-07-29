"""Karman Vortex Street
Air flow around a static cylinder.
Vortices start appearing after a couple of hundred steps.
"""
from phi.flow import *  # minimal dependencies
from phi.physics._effect import FieldEffect
# from phi.torch.flow import *

# from phi.tf.flow import *
# from phi.jax.flow import *
from src.env.phiflow.navier_stokes import NavierStokes

# SPEED = vis.control(2.)
SPEED = 2.0
velocity = StaggeredGrid((SPEED, 0), extrapolation.BOUNDARY, x=128, y=64, bounds=Box(x=128, y=64))
CYLINDER = Obstacle(geom.infinite_cylinder(x=15, y=32, radius=4, inf_dim=None))
BOUNDARY_MASK = StaggeredGrid(Box(x=(-INF, 0.5), y=None), velocity.extrapolation, velocity.bounds, velocity.resolution)
pressure = None
dt = 1.0


def officialGaussianClash(x):
    batch_size = 32
    leftloc = np.random.uniform(0.2, 0.4)
    leftamp = np.random.uniform(0, 3)
    leftsig = np.random.uniform(0.05, 0.15)
    rightloc = np.random.uniform(0.6, 0.8)
    rightamp = np.random.uniform(-3, 0)
    rightsig = np.random.uniform(0.05, 0.15)
    left = leftamp * math.exp(-0.5 * (x.vector[0] - leftloc) ** 2 / leftsig ** 2)
    right = rightamp * math.exp(-0.5 * (x.vector[0] - rightloc) ** 2 / rightsig ** 2)
    result = left + right
    return result


def officialGaussianForce(x):
    batch_size = 1
    for i in range(len(x.shape)):
        batch_size *= int(x.shape[i])
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    reshap = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    result = tensor(amp.reshape(reshap), x.shape) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc.reshape(reshap), x.shape)) ** 2 / tensor(sig.reshape(reshap), x.shape) ** 2)
    return result


def apply_effect_mask(x: math.Tensor):
    for i in range(len(x)):
        if i < len(x) / 3:
            for j in range(len(x[i])):
                if j > len(x[i]) / 2:
                    for k in range(len(x[i][j])):
                        x[i][j][k] = -0.5
                else:
                    for k in range(len(x[i][j])):
                        x[i][j][k] = 0.0
        else:
            for j in range(len(x[i])):
                for k in range(len(x[i][j])):
                    x[i][j][k] = 0.0

    return x
def burgers_rkstiff_function(x):
    # x._native = apply_effect_mask(x._native)
    u0 = math.exp(-10 * math.sin(x / 2) ** 2)
    return u0


effect_grid = CenteredGrid(
    officialGaussianForce, extrapolation=extrapolation.ZERO,
    resolution=velocity.resolution)
effect1 = FieldEffect(effect_grid, ['v1'])
velocity_effects = ()

physics = NavierStokes(speed=SPEED)
for i in view('vorticity,velocity,pressure',
              play=True, namespace=globals()).range():
    velocity, pressure, vorticity = physics.step(
        v=velocity, dt=dt, boundary_mask=BOUNDARY_MASK,
        pressure=pressure, obstacles=[CYLINDER],
        velocity_effects=velocity_effects)
