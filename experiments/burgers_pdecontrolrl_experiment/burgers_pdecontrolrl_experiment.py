from matplotlib import pyplot as plt
from phi.flow import *
from tqdm import tqdm

from src.util.burgers_util import GaussianClash, GaussianForce
from src.experiment import BurgersTrainingExpr


def main():
    domain = Domain([32], box=box[0:1])
    viscosity = 0.003
    step_count = 32
    dt = 0.03
    diffusion_substeps = 1

    data_path = 'forced-burgers-clash'
    scene_count = 500
    batch_size = 10

    train_range = range(200, 1000)
    val_range = range(100, 200)
    test_range = range(0, 100)

    for batch_index in range(scene_count // batch_size):
        scene = Scene.create(data_path, count=batch_size)
        print(scene)
        world = World()
        u0 = BurgersVelocity(
            domain,
            velocity=GaussianClash(batch_size),
            viscosity=viscosity,
            batch_size=batch_size,
            name='burgers'
        )
        u = world.add(u0, physics=Burgers(diffusion_substeps=diffusion_substeps))
        force = world.add(FieldEffect(GaussianForce(batch_size), ['velocity']))
        scene.write(world.state, frame=0)
        for frame in range(1, step_count + 1):
            world.step(dt=dt)
            scene.write(world.state, frame=frame)

    n_envs = 1  # On how many environments to train in parallel, load balancing
    final_reward_factor = step_count  # How hard to punish the agent for not reaching the goal if that is the case
    steps_per_rollout = step_count * 1  # How many steps to collect per environment between agent updates
    n_epochs = 100  # How many epochs to perform during agent update
    learning_rate = 1e-4  # Learning rate for agent updates
    rl_batch_size = n_envs * step_count  # Batch size for agent updates

    rl_trainer = BurgersTrainingExpr(
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
        # test_path=None, # data_path,
        test_range=None,  # test_range,
    )

    rl_trainer.train(n_rollouts=2, save_freq=10)
    rl_trainer.render()
    plt.show()


if __name__ == '__main__':
    main()
