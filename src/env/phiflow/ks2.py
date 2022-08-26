from phi.flow import *
from phi.physics._physics import Physics


class KuramotoSivashinsky(Physics):

    def __init__(self, k):
        Physics.__init__(self)
        self.k = k

    def N_(self, u):
        u = math.ifft(u)
        return 0.5j * self.k * math.fft(u * u)

    def P(self, u, e_Ldt, dt, L):
        u = math.fft(u)
        an = u * e_Ldt + self.N_(u) * math.where(L == 0, 0.5, (e_Ldt - 1) / L)
        u1 = an + (self.N_(an) - self.N_(u)) * math.where(L == 0, 0.25, (e_Ldt - 1 - L * dt) / (L ** 2 * dt))
        return math.real(math.ifft(u1))

    def step(self, u, dt=1.0, e_Ldt=1.0, **dependent_states):

        grad = field.spatial_gradient(u)
        laplace = field.laplace(u)
        laplace2 = field.laplace(field.laplace(u))
        du_dt = -laplace - laplace2 - 0.5 * grad ** 2
        result = u + dt * du_dt
        result -= math.mean(result.data)
        return result