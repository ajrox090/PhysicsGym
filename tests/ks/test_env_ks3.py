""" Kuramoto–Sivashinsky Equation
Simulates the KS equation in one dimension.
Supports PyTorch, TensorFlow and Jax; select backend via import statement.
"""
from phi.torch.flow import *
from tqdm import tqdm

# from phi.tf.flow import *
# from phi.jax.flow import *

from src.util.ks_util import ks_initial
from src.env.physics.ks3 import KuramotoSivashinsky

dt = .50

ks = KuramotoSivashinsky()
a = CenteredGrid(ks_initial, x=128, bounds=Box(x=22))
# trajectory = [a]
# for i in view('trajectory', play=False, namespace=globals()).range():
for i in tqdm(range(1000)):
    # print(f"Step {i}: max value {trajectory[-1].values.max}")
    a = ks.step(a.vecotr['x'], dt=dt)
    # trajectory.append(a.vector['x'])
    # if i % 10 == 0:
    #     vis.show(a, aspect='auto', size=(8, 6))
trajectory = field.stack(ks.trajectory, spatial('time'), Box(time=1000 * dt))

vis.show(trajectory.vector[0], aspect='auto', size=(8, 6))
