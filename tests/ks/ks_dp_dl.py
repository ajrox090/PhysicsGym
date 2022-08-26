from phi.torch.flow import *
import numpy as np
from tqdm import tqdm

print(f'CUDA Available? {torch.cuda.is_available()}')

X = 8
x = math.linspace(0, 8, 48, dim=spatial('xs'))
random_sign = math.sign(math.random_uniform(batch(b=6), low=-1, high=1))
alpha = math.random_uniform(batch(b=6), low=-8, high=8)

u_0 = math.cos(2 * x) + 0.1 * random_sign * math.cos(2 * math.pi * x / X) * (1 - alpha * math.sin(2 * math.pi * x / X))
# u_0 = math.cos(2*x) + 0.1 * math.cos(2 * math.pi * x/X) * (1 - math.sin(2 * math.pi * x/X))

# Precomputing Constants
k = math.fftfreq(spatial(xs=int(x.shape[0])), dx=1) * 2 * math.pi
k = math.unstack(k, 'vector')[0]

# This is what the above code does
# k1 = math.linspace(-23,-1,23,spatial('xs')) * 2 * math.pi / 48
# k2 = math.linspace(0,23,24,spatial('xs')) * 2 * math.pi / 48
# k3 = math.linspace(24,24,1, spatial('xs')) * 2 * math.pi / 48
# k = math.concat([k2,k3,k1], 'xs')

L = k ** 2 - k ** 4
dt = 0.5
e_Ldt = math.exp(L * dt)


def N_(u):
    u = math.ifft(u)
    return -0.5j * k * math.fft(u * u)


def P(u):
    u = math.fft(u)
    an = u * e_Ldt + N_(u) * math.where(L == 0, 0.5, (e_Ldt - 1) / L)
    u1 = an + (N_(an) - N_(u)) * math.where(L == 0, 0.25, (e_Ldt - 1 - L * dt) / (L ** 2 * dt))
    return math.real(math.ifft(u1))


u = u_0
u_solution = math.expand(u, spatial('t'))

for i in tqdm(range(100)):
    u = P(u)
    u_solution = math.concat([u_solution, math.expand(u, spatial('t'))], dim='t')

# Plotting batch = 0 solution
vis.show(CenteredGrid(math.real(u_solution.b[0]), 0, Box(xs=100, t=500)))
