from matplotlib import pyplot as plt
from phi.flow import *
from tqdm import tqdm

from src.util.burgers_util import GaussianClash, GaussianForce
from src.experiment import BurgersTrainingExpr


def main():
    N = 128
    domain = Domain([N], box=box[-1:1])
    viscosity = 0.03
    # viscosity = 0.01/(N*np.pi)
    step_count = 32
    # dt = 1./step_count
    dt = 0.01
    diffusion_substeps = 1

    n_envs = 1  # On how many environments to train in parallel, load balancing
    final_reward_factor = step_count  # How hard to punish the agent for not reaching the goal if that is the case
    steps_per_rollout = step_count * 1  # How many steps to collect per environment between agent updates
    n_epochs = 100  # How many epochs to perform during agent update
    learning_rate = 1e-4  # Learning rate for agent updates
    rl_batch_size = n_envs * step_count  # Batch size for agent updates

    rl_trainer = BurgersTrainingExpr(
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
    # rl_trainer.reset_env()
    # rl_trainer.step_env()
    # rl_trainer.show_state()
    rl_trainer.train(n_rollouts=2, save_freq=10)
    # rl_trainer.visualize(step_count, N)
    rl_trainer.show_state()
    # rl_trainer.plot()
    # plt.show()
    # rl_trainer.render_env(mode='live')
    # plt.show()


if __name__ == '__main__':
    main()
