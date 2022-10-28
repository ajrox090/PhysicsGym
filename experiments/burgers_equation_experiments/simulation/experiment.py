from phi.flow import *
from scipy.stats import norm


def gauss(x):
    xshape = x.shape
    leftloc = math.random_uniform(xshape, low=0.2, high=0.4)
    leftamp = math.random_uniform(xshape, low=0, high=3)
    leftsig = math.random_uniform(xshape, low=0.05, high=0.15)
    rightloc = math.random_uniform(xshape, low=0.6, high=0.8)
    rightamp = math.random_uniform(xshape, low=-3, high=0)
    rightsig = math.random_uniform(xshape, low=0.05, high=0.15)

    left = leftamp * math.exp(-0.5 * (x - leftloc) ** 2 / leftsig ** 2)
    right = rightamp * math.exp(-0.5 * (x - rightloc) ** 2 / rightsig ** 2)
    result = left + right
    return result


N = 128
count = 5
STEPS = 32
DX = 2. / N
DT = 1. / STEPS
NU = 0.01 / (N * np.pi)
step_count = 32

domain_dict = dict(x=N, bounds=Box[0:1],
                   extrapolation=extrapolation.PERIODIC)
# initialization of velocities, cell centers of a CenteredGrid have DX/2 offsets for linspace()
INITIAL_NUMPY = np.asarray(
    [-np.sin(np.pi * x) + np.cos(np.pi * x) for x in np.linspace(-1 + DX / 2, 1 - DX / 2, N)])  # 1D numpy array

INITIAL = math.tensor(INITIAL_NUMPY, spatial('x'))  # convert to phiflow tensor


def NORMAL(x):
    result = None
    if len(x.shape) == 2:
        result = tensor(norm.pdf(x.native("vector,x")[0], 0.2, 0.2), x.shape[0])
    elif len(x.shape) == 3:
        result = tensor(norm.pdf(x.native("vector,x,y")[0], 0.2, 0.2), x.shape[:2])
    return result


velocity = CenteredGrid(NORMAL, extrapolation.PERIODIC, x=N, y=N, bounds=Box(x=(-1, 1), y=(-1, 1)))
velocities = [velocity]
age = 0.
for i in range(STEPS):
    v1 = diffuse.explicit(velocities[-1], NU, DT)
    v2 = advect.semi_lagrangian(v1, v1, DT)
    age += DT
    velocities.append(v2)

print("New velocity content at t={}: {}".format(age, velocities[-1].values.numpy('x,y,vector')[0:5]))
# get "velocity.values" from each phiflow state with a channel dimensions, i.e. "vector"
vels = [v.values.numpy('x,y,vector') for v in velocities]  # gives a list of 2D arrays

for vel in velocities:
    vis.show(vel)
# import pylab
#
# fig = pylab.figure().gca()
# # pylab.grid()
# fig.plot(np.linspace(-1, 1, len(vels[0].flatten())), vels[0].flatten(), lw=1, color='blue', label="t=0")
# fig.plot(np.linspace(-1, 1, len(vels[10].flatten())), vels[10].flatten(), lw=1, color='green', label="t=0.3125")
# fig.plot(np.linspace(-1, 1, len(vels[20].flatten())), vels[20].flatten(), lw=1, color='cyan', label="t=0.625")
# fig.plot(np.linspace(-1, 1, len(vels[32].flatten())), vels[32].flatten(), lw=1, color='purple', label="t=1")
# pylab.xlabel('x')
# pylab.ylabel('u')
# pylab.legend()
# pylab.show()
