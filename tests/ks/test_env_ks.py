import math

from phi.torch.flow import *
from phi.torch.nets import adam, update_weights
from tqdm import tqdm

from src.env.physics.ks import KuramotoSivashinsky
from src.util.ks_util import ks_initial, ks_initial2

#
N = 128
step_count = 100
dt = 0.01
u = CenteredGrid(ks_initial, x=N, extrapolation=extrapolation.PERIODIC, bounds=Box(x=22))
physics = KuramotoSivashinsky()

# for _ in view('u', play=True, framerate=10, namespace=globals()).range():
trajectory = [u.vector['x']]
for i in range(step_count):
    u = physics.step(u, dt=dt)
    trajectory.append(u.vector['x'])
temp_t = field.stack(trajectory, spatial('time'),
                     Box(time=step_count * dt))
vis.show(temp_t.vector[0], aspect='auto', size=(8, 6))

# N = 48
# X = 8
# x = math.linspace(0, 8, N, dim=spatial('xs'))
# random_sign = math.sign(math.random_uniform(batch(b=6), low=-1, high=1))
# alpha = math.random_uniform(batch(b=6), low=-8, high=8)
#
# u = math.cos(2 * x) + 0.1 * random_sign * math.cos(2 * math.pi * x / X) * (1 - alpha * math.sin(2 * math.pi * x / X))
#
# k = math.fftfreq(spatial(xs=int(x.shape[0])), dx=1) * 2 * math.pi
# k = math.unstack(k, 'vector')[0]
#
# L = k ** 2 - k ** 4
# dt = 0.5
# e_Ldt = math.exp(L * dt)
#
#
# def P(u):
#     def N_(u):
#         u = math.ifft(u)
#         return -0.5j * k * math.fft(u * u)
#
#     u = math.fft(u)
#     an = u * e_Ldt + N_(u) * math.where(L == 0, 0.5, (e_Ldt - 1) / L)
#     u1 = an + (N_(an) - N_(u)) * math.where(L == 0, 0.25, (e_Ldt - 1 - L * dt) / (L ** 2 * dt))
#     return math.real(math.ifft(u1))
#
#
# # solve for n time steps
# u_solution = math.expand(u, spatial('t'))
# for i in tqdm(range(2000)):
#     u = P(u)
#     u_solution = math.concat([u_solution, math.expand(u, spatial('t'))], dim='t')  # add for later
#
# # just see two solutions
# vis.show(CenteredGrid(math.real(u_solution.b[0]), 0, Box(xs=100, t=500)))
# vis.show(CenteredGrid(math.real(u_solution.b[2]), 0, Box(xs=100, t=500)))
#
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class ConvResNet(nn.Module):
#     def __init__(self):
#         super(ConvResNet, self).__init__()
#
#         self.conv1 = nn.Conv1d(1, 4, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn1 = nn.BatchNorm1d(4)
#
#         self.conv2 = nn.Conv1d(4, 16, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn2 = nn.BatchNorm1d(16)
#
#         self.conv_short1 = nn.Conv1d(1, 16, 1, 1)
#
#         self.conv3 = nn.Conv1d(16, 32, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn3 = nn.BatchNorm1d(32)
#
#         self.conv4 = nn.Conv1d(32, 32, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn4 = nn.BatchNorm1d(32)
#
#         self.conv_short2 = nn.Conv1d(16, 32, 1, 1)
#
#         self.conv5 = nn.Conv1d(32, 32, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn5 = nn.BatchNorm1d(32)
#
#         self.conv6 = nn.Conv1d(32, 32, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn6 = nn.BatchNorm1d(32)
#
#         self.conv_short3 = nn.Conv1d(32, 32, 1, 1)
#
#         self.conv7 = nn.Conv1d(32, 16, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn7 = nn.BatchNorm1d(16)
#
#         self.conv8 = nn.Conv1d(16, 8, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn8 = nn.BatchNorm1d(8)
#
#         self.conv_short4 = nn.Conv1d(32, 8, 1, 1)
#
#         self.conv9 = nn.Conv1d(8, 8, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn9 = nn.BatchNorm1d(8)
#
#         self.conv10 = nn.Conv1d(8, 4, 3, stride=1, padding=1, padding_mode='circular')
#         # self.bn10 = nn.BatchNorm1d(4)
#
#         self.conv_short5 = nn.Conv1d(8, 4, 1, 1)
#
#         self.conv11 = nn.Conv1d(4, 1, 3, stride=1, padding=1, padding_mode='circular')
#
#     def forward(self, x):
#         '''
#         x1 = F.relu(self.bn1(self.conv1(x)))
#         x1 = F.relu(self.bn2(self.conv2(x1)))
#         x1 = x1 + self.conv_short1(x)
#
#         x2 = F.relu(self.bn3(self.conv3(x1)))
#         x2 = F.relu(self.bn4(self.conv4(x2)))
#         x2 = x2 + self.conv_short2(x1)
#
#         x3 = F.relu(self.bn5(self.conv5(x2)))
#         x3 = F.relu(self.bn6(self.conv6(x3)))
#         x3 = x3 + self.conv_short3(x2)
#
#         x4 = F.relu(self.bn7(self.conv7(x3)))
#         x4 = F.relu(self.bn8(self.conv8(x4)))
#         x4 = x4 + self.conv_short4(x3)
#
#         x5 = F.relu(self.bn9(self.conv9(x4)))
#         x5 = F.relu(self.bn10(self.conv10(x5)))
#         x5 = x5 + self.conv_short5(x4)
#         '''
#         x1 = F.relu(self.conv1(x))
#         x1 = F.relu(self.conv2(x1))
#         x1 = x1 + self.conv_short1(x)
#
#         x2 = F.relu(self.conv3(x1))
#         x2 = F.relu(self.conv4(x2))
#         x2 = x2 + self.conv_short2(x1)
#
#         x3 = F.relu(self.conv5(x2))
#         x3 = F.relu(self.conv6(x3))
#         x3 = x3 + self.conv_short3(x2)
#
#         x4 = F.relu(self.conv7(x3))
#         x4 = F.relu(self.conv8(x4))
#         x4 = x4 + self.conv_short4(x3)
#
#         x5 = F.relu(self.conv9(x4))
#         x5 = F.relu(self.conv10(x5))
#         x5 = x5 + self.conv_short5(x4)
#
#         out = self.conv11(x5)
#
#         return out
#
#
# # Test Data
# alpha_test = math.random_uniform(batch(b=1), low=-1, high=1) * 8
# u_test_0 = math.cos(2 * x) - 0.1 * math.cos(2 * math.pi * x / X) * (1 - alpha_test * math.sin(2 * math.pi * x / X))
# model_dp = ConvResNet().to('cuda')
#
#
# def norm(v):
#     return math.sqrt(2 * math.l2_loss(v))
#
#
# def loss_func_dp(u):
#     loss = 0
#     for i in range(5):
#         u_pred = math.native_call(model_dp, u)
#         loss += math.l2_loss(u_pred - P(u)) / (1 + math.l2_loss(u_pred))
#         u = u_pred
#     return math.sum(loss, 'b')
#
#
# num_epochs = 1
# num_time_steps = 2000
# u_dp_solution = math.expand(u, spatial('t'))
# optimizer = adam(model_dp, learning_rate=1e-3)
# for epoch in range(1, num_epochs + 1):
#     for step in range(num_time_steps):
#         loss = update_weights(model_dp, optimizer, loss_func_dp, u)
#         # u1 = math.native_call(model_dp, u)
#         u = P(u)
#         if step % 1000 == 0:
#             print(f'Epoch : {epoch}, step : {step}, Loss : {loss}')
#         u_dp_solution = math.concat([u_dp_solution, math.expand(u, spatial('t'))], 't')
# u_test = u_test_0
#
# # Loss_L2 : Between simulated trajectory and predicted trajectory
# Loss_L2 = math.expand(math.l2_loss(math.native_call(model_dp, u_test) - P(u_test)), instance('i'))
#
# # Loss_P : Between function P and neural network fnn where un is predicted trajectory
# Loss_P = Loss_L2
#
# u_pred = math.native_call(model_dp, u_test)
# u_test = P(u_test)
# u_pred_solution = math.expand(u_pred, spatial('t'))
# u_solver_solution = math.expand(u_test, spatial('t'))
# for step in range(500):
#
#     Loss_P = math.concat(
#         [Loss_P, math.expand(math.l2_loss(math.native_call(model_dp, u_pred) - P(u_pred)), instance('i'))], 'i')
#     Loss_L2 = math.concat(
#         [Loss_L2, math.expand(math.l2_loss(math.native_call(model_dp, u_pred) - P(u_test)), instance('i'))], 'i')
#
#     u_pred = math.native_call(model_dp, u_pred)
#     u_test = P(u_test)
#
#     u_pred_solution = math.concat([u_pred_solution, math.expand(u_pred, spatial('t'))], 't')
#     u_solver_solution = math.concat([u_solver_solution, math.expand(u_test, spatial('t'))], 't')
#
#     if step % 20 == 0:
#         print(f'Step: {step}, Loss_L2 : {Loss_L2.i[-1]}, Loss_Physics : {Loss_P.i[-1]}')
# vis.show(CenteredGrid(math.real(u_pred_solution.b[0]), 0, Box(xs=100, t=500)))
# vis.show(CenteredGrid(math.real(u_solver_solution.b[0]), 0, Box(xs=100, t=500)))
