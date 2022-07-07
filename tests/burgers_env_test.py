import unittest

from phi.flow import *

from src.experiment import BurgersTrainingExpr


def setup_rltrainer():
    N = 32
    domain = Domain([N], box=box[0:1])
    viscosity = 0.01 / (N * np.pi)
    step_count = 32
    dt = 1. / step_count
    diffusion_substeps = 1

    n_envs = 1  # On how many environments to train in parallel, load balancing
    steps_per_rollout = step_count * 1  # How many steps to collect per environment between agent updates
    n_epochs = 100  # How many epochs to perform during agent update
    learning_rate = 1e-4  # Learning rate for agent updates
    rl_batch_size = n_envs * step_count  # Batch size for agent updates

    return BurgersTrainingExpr(
        N,
        path='networks/rl-models/time_bench',
        domain=domain,
        viscosity=viscosity,
        step_count=step_count,
        dt=dt,
        diffusion_substeps=diffusion_substeps,
        n_envs=n_envs,
        steps_per_rollout=steps_per_rollout,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=rl_batch_size,
        test_range=None,  # test_range,
    )


class MyTestCase(unittest.TestCase):
    def test_show_vels(self):
        rl_trainer = setup_rltrainer()

        rl_trainer.train(n_rollouts=2, save_freq=10)
        rl_trainer.show_vels()


if __name__ == '__main__':
    unittest.main()
