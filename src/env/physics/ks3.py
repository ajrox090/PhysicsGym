from phi.flow import *
from phi.physics._effect import effect_applied
from phi.physics._physics import Physics, StateDependency


class KuramotoSivashinsky(Physics):

    def __init__(self):
        Physics.__init__(self, [StateDependency('effects', 'velocity_effect', blocking=True)])
        self.trajectory = []

    def step(self, u, dt=1.0, effects=()):
        # --- Operators in Fourier space ---
        frequencies = math.fftfreq(u.resolution) / u.dx
        lin_op = frequencies ** 2 - (
                1j * frequencies) ** 4
        # Fourier operator for linear terms. You'd think that 1j**4 == 1 but
        # apparently the rounding errors have a major effect here even with FP64...
        inv_lin_op = math.divide_no_nan(1, lin_op)  # Removes f=0 component but there is no noticeable difference
        exp_lin_op = math.exp(lin_op * dt)  # time evolution operator for linear terms in Fourier space
        # --- RK2 for non-linear terms, exponential time-stepping for linear terms ---
        non_lin_current = -0.5j * frequencies * math.fft(u.values ** 2)
        u_intermediate = exp_lin_op * math.fft(u.values) + non_lin_current * (
                exp_lin_op - 1) * inv_lin_op  # intermediate for RK2
        non_lin_intermediate = -0.5j * frequencies * math.fft(math.ifft(u_intermediate).real ** 2)
        u_new = u_intermediate + (non_lin_intermediate - non_lin_current) * (exp_lin_op - 1 - lin_op * dt) * (
                1 / dt * inv_lin_op ** 2)
        final_u = u.with_values(math.ifft(u_new).real)
        for effect in effects:
            if type(effect) is not tuple:
                final_u = effect_applied(effect, final_u, dt)
        self.trajectory.append(final_u)
        return final_u
