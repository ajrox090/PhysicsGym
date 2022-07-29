from phi.flow import *
from src.env.phiflow.ks import KuramotoSivashinsky

N = 32
domain_dict = dict(x=N, extrapolation=extrapolation.PERIODIC)
step_count = 1
dt = 0.01


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
    batch_size = 32
    loc = np.random.uniform(0.4, 0.6, batch_size)
    amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
    sig = np.random.uniform(0.1, 0.4, batch_size)
    result = tensor(amp, x.shape[0]) * math.exp(
        -0.5 * (x.x.tensor - tensor(loc, x.shape[0])) ** 2 / tensor(sig, x.shape[0]) ** 2)
    return result


def burgers_rkstiff_function(x):
    u0 = math.exp(-10 * math.sin(x / 2) ** 2)
    return u0


u = CenteredGrid(Noise(scale=5), x=128,
                 extrapolation=extrapolation.PERIODIC,
                 bounds=Box(x=128))

physics = KuramotoSivashinsky()

for _ in view('u', play=True, framerate=5, namespace=globals()).range():
# for _ in range(5):
    u = physics.step(u, dt=0.1)
    # vis.show(u)
