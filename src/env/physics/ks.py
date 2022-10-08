from phi.flow import *
from phi.physics._effect import effect_applied
from phi.physics._physics import Physics


class KuramotoSivashinsky(Physics):

    def __init__(self):
        Physics.__init__(self)

    def step(self, u, dt=1.0, effects=()):
        # assert isinstance(u, CenteredGrid)
        grad = field.spatial_gradient(u)
        laplace = field.laplace(u)
        laplace2 = field.laplace(field.laplace(u))
        du_dt = -laplace - laplace2 - 0.5 * grad ** 2
        result = u + dt * du_dt
        result -= math.mean(result.data)

        for effect in effects:
            result = effect_applied(effect, result, dt)
        return result
