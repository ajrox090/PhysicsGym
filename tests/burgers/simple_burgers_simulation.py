""" Simulate Burgers' Equation
Simple advection-diffusion equation.
"""
from phi.flow import *
from scipy.stats import norm

from src.env.PhysicsGym import PhysicsGym


def simpleNormalDistribution(x):
    result = tensor(norm.pdf(x.native("vector,x,y")[0], 0.2, 0.5), x.shape[:2])
    return result


velocity = CenteredGrid(simpleNormalDistribution, extrapolation.PERIODIC, x=64, y=64,
                        bounds=Box[0:1, 0:1])


# @jit_compile  # for PyTorch, TensorFlow and Jax
def burgers_step(v, dt=1.):
    v = diffuse.explicit(v, 0.1, dt=dt)
    v = advect.semi_lagrangian(v, v, dt=dt)
    return v


velocity = burgers_step(velocity)
vis.show(velocity)