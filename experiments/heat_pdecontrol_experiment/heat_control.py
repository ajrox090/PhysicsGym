from phi.flow import *
from matplotlib import pyplot as plt
from src.experiment import HeatTrainingExper


def main():
    domain = Domain([32, ], box=box[0:1])
    diffusivity = 0.01
    step_count = 2
    dt = 0.03

    n_envs = 1  # On how many environments to train in parallel, load balancing
    final_reward_factor = step_count  # How hard to punish the agent for not reaching the goal if that is the case
    steps_per_rollout = step_count * 1  # How many steps to collect per environment between agent updates
    n_epochs = 100  # How many epochs to perform during agent update
    learning_rate = 1e-4  # Learning rate for agent updates
    rl_batch_size = n_envs * step_count  # Batch size for agent updates

    rl_trainer = HeatTrainingExper(
        path='networks/rl-models/time_bench',
        domain=domain,
        diffusivity=diffusivity,
        step_count=step_count,
        dt=dt,
        n_envs=n_envs,
        steps_per_rollout=steps_per_rollout,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=rl_batch_size,
        test_range=None,  # test_range,
    )

    rl_trainer.train(n_rollouts=2, save_freq=10)
    rl_trainer.render_env()
    plt.show()


if __name__ == '__main__':
    main()
