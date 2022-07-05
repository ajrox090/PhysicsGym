import math
import os
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common import logger

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.env.burgers_env import BurgersEnv
from src.env.heat_invader_env import HeatInvaderEnv

from src.policy import CustomActorCriticPolicy
from src.env.burgers_fixedset_env import BurgersFixedSetEnv
from src.networks import RES_UNET, CNN_FUNNEL
from src.vis.monitor_env import VecMonitor


class ExperimentFolder:
    agent_filename = 'agent'
    monitor_filename = 'monitor.csv'
    kwargs_filename = 'kwargs'
    tensorboard_filename = 'tensorboard-log'

    def __init__(self, path):
        self.store_path = path
        self.agent_path = os.path.join(self.store_path, self.agent_filename)
        self.monitor_path = os.path.join(self.store_path, self.monitor_filename)
        self.kwargs_path = os.path.join(self.store_path, self.kwargs_filename)
        self.tensorboard_path = os.path.join(self.store_path, self.tensorboard_filename)

        if not self.can_be_loaded:
            os.makedirs(self.agent_path)

    @property
    def can_be_loaded(self):
        return os.path.exists(self.agent_path)

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    def store_agent_only(self, agent):
        print('Storing agent to disk...')
        agent.save(self.agent_path)

    def store(self, agent, env_kwargs, agent_kwargs):
        print('Storing agent and hyperparameters to disk...')
        kwargs = self._group_kwargs(env_kwargs, agent_kwargs)
        agent.save(self.agent_path)
        with open(self.kwargs_path, 'wb') as kwargs_file:
            pickle.dump(kwargs, kwargs_file)

    def get(self, env_cls, env_kwargs, agent_kwargs):
        print('Tensorboard log path: %s' % self.tensorboard_path)
        if not self.can_be_loaded:  # disabled temporariliy
            print('Loading existing agent from %s' % (self.agent_path + '.zip'))
            return self._load(env_cls, env_kwargs, agent_kwargs)
        else:
            print('Creating new agent...')
            return self._create(env_cls, env_kwargs, agent_kwargs)

    def get_monitor_table(self):
        return pd.read_csv(self.monitor_path, skiprows=[0])

    def get_tensorboard_scalar(self, scalar_name):
        path_template = os.path.join(self.tensorboard_path, 'training_phase_%i')
        # Compatibility with other naming scheme, TODO not good code, needs another revision
        if not os.path.exists(path_template % 0):
            path_template = os.path.join(self.tensorboard_path, 'PPO_%i')
        run_idx = 0
        wall_times, timesteps, scalar_values = [], [], []
        while os.path.exists(path_template % run_idx):
            event_accumulator = EventAccumulator(path_template % run_idx)
            event_accumulator.Reload()

            new_wall_times, new_timesteps, new_scalar_values = zip(*event_accumulator.Scalars(scalar_name))

            # To chain multiple runs together, the time inbetween has to be left out
            prev_run_wall_time = 0 if len(wall_times) == 0 else wall_times[-1]
            # Iterations have to be continuous even when having multiple runs
            prev_run_timesteps = 0 if len(timesteps) == 0 else timesteps[-1]

            wall_times += [prev_run_wall_time + wt - new_wall_times[0] for wt in new_wall_times]
            timesteps += [prev_run_timesteps + it - new_timesteps[0] for it in new_timesteps]
            scalar_values += new_scalar_values

            run_idx += 1

        return wall_times, timesteps, scalar_values

    def get_monitor_scalar(self, scalar_name):
        table = pd.read_csv(self.monitor_path, skiprows=[0])
        wall_times = list(table['t'])
        iterations = [i for i in range(len(wall_times))]
        scalar_values = list(table[scalar_name])

        # Make wall times of multiple runs monotonic:
        base_time = 0
        monotonic_wall_times = [wall_times[0]]
        for i in range(1, len(wall_times)):
            if wall_times[i] < wall_times[i - 1]:
                base_time = monotonic_wall_times[i - 1]
            monotonic_wall_times.append(wall_times[i] + base_time)

        return monotonic_wall_times, iterations, scalar_values

    def _create(self, env_cls, env_kwargs, agent_kwargs):
        env = self._build_env(env_cls, env_kwargs, agent_kwargs['n_steps'])
        agent = PPO(env=env, tensorboard_log=self.tensorboard_path, **agent_kwargs)
        return agent, env

    def _load(self, env_cls, env_kwargs, agent_kwargs):
        with open(self.kwargs_path, 'rb') as kwargs_file:
            kwargs = pickle.load(kwargs_file)
        kwargs['env'].update(env_kwargs)
        kwargs['agent'].update(agent_kwargs)
        env = self._build_env(env_cls, kwargs['env'], kwargs['agent']['n_steps'])
        agent = PPO.load(path=self.agent_path, env=env, tensorboard_log=self.tensorboard_path, **kwargs['agent'])
        return agent, env

    def _build_env(self, env_cls, env_kwargs, rollout_size):
        env = env_cls(**env_kwargs)
        return VecMonitor(env, rollout_size, self.monitor_path, info_keywords=('rew_unnormalized', 'forces'))

    @staticmethod
    def _group_kwargs(env_kwargs, agent_kwargs):
        return dict(
            env=env_kwargs,
            agent=agent_kwargs,
        )


class Experiment:
    def __init__(self, path, env_cls, env_kwargs, agent_kwargs, steps_per_rollout, num_envs=1):
        self.folder = ExperimentFolder(path)
        self.agent, self.env = self.folder.get(env_cls, env_kwargs, agent_kwargs)
        self.steps_per_rollout = steps_per_rollout
        self.num_envs = num_envs

        var = lambda _: self.folder.store(self.agent, env_kwargs, agent_kwargs)

    def train(self, n_rollouts, save_freq):
        self.agent.learn(total_timesteps=n_rollouts * self.steps_per_rollout * self.num_envs,
                         tb_log_name="training_phase")

    def plot(self):
        monitor_table = self.folder.get_monitor_table()
        avg_rew = monitor_table['rew_unnormalized']
        plt.title('Reward unnormalized')
        return plt.plot(avg_rew)

    def visualize(self, step_count=32, N=128):
        STEPS = step_count
        t0gt = np.asarray([[-math.sin(np.pi * x) * 1.] for x in np.linspace(-1, 1, N)])
        assert len(self.env.venv.vis_list) > 0
        vels = self.env.venv.vis_list
        vel_resim = [x.velocity.data for x in vels]
        fig = plt.figure().gca()
        pltx = np.linspace(-1, 1, len(vel_resim[0].flatten()))
        fig.plot(pltx, vel_resim[0].flatten(), lw=2, color='blue', label="t=0")
        fig.plot(pltx, vel_resim[STEPS//4].flatten(), lw=2, color='green', label="t=0.125")
        fig.plot(pltx, vel_resim[STEPS // 2].flatten(), lw=2, color='cyan', label="t=0.25")
        fig.plot(pltx, vel_resim[STEPS - 1].flatten(), lw=2, color='purple', label="t=0.5")
        fig.plot(pltx, t0gt, lw=2, color='gray', label="t=0 Reference")
        # optionally show GT, compare to ˓→blue
        plt.title("Resimulated u from solution at t=0")
        plt.xlabel('x')
        plt.ylabel('u')
        plt.legend()
        plt.show()

    def reset_env(self):
        return self.env.reset()

    def predict(self, obs, deterministic=True):
        act, _ = self.agent.predict(obs, deterministic=deterministic)
        return act

    def step_env(self, act):
        return self.env.step(act)

    def render_env(self, mode: str):
        assert isinstance(self.env, VecEnv)
        self.env.render(mode=mode)


class BurgersTrainingExpr(Experiment):
    def __init__(
            self,
            path,
            domain,
            viscosity,
            step_count,
            dt,
            diffusion_substeps,
            n_envs,
            steps_per_rollout,
            n_epochs,
            learning_rate,
            batch_size,
            data_path=None,
            val_range=range(100, 200),
            test_range=range(100),
    ):
        env_kwargs = dict(
            num_envs=n_envs,
            step_count=step_count,
            domain=domain,
            dt=dt,
            viscosity=viscosity,
            diffusion_substeps=diffusion_substeps,
            exp_name=path,
        )

        evaluation_env_kwargs = {k: env_kwargs[k] for k in env_kwargs if k != 'num_envs'}

        # Only add a fresh running mean to new experiments
        if not ExperimentFolder.exists(path):
            env_kwargs['reward_rms'] = RunningMeanStd()

        agent_kwargs = dict(
            verbose=0,
            policy=CustomActorCriticPolicy,
            policy_kwargs=dict(
                pi_net=RES_UNET,
                vf_net=CNN_FUNNEL,
                vf_latent_dim=16,
                pi_kwargs=dict(
                    sizes=[4, 8, 16, 16, 16]
                ),
                vf_kwargs=dict(
                    sizes=[4, 8, 16, 16, 16]
                ),
            ),
            n_steps=steps_per_rollout,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        super().__init__(path, BurgersEnv, env_kwargs, agent_kwargs, steps_per_rollout, n_envs)


class HeatTrainingExper(Experiment):
    def __init__(
            self,
            path,
            domain,
            diffusivity,
            step_count,
            dt,
            n_envs,
            steps_per_rollout,
            n_epochs,
            learning_rate,
            batch_size,
            data_path=None,
            val_range=range(100, 200),
            test_range=range(100),
    ):
        env_kwargs = dict(
            num_envs=n_envs,
            step_count=step_count,
            domain=domain,
            dt=dt,
            diffusivity=diffusivity,
            exp_name=path,
        )

        evaluation_env_kwargs = {k: env_kwargs[k] for k in env_kwargs if k != 'num_envs'}

        # Only add a fresh running mean to new experiments
        if not ExperimentFolder.exists(path):
            env_kwargs['reward_rms'] = RunningMeanStd()

        agent_kwargs = dict(
            verbose=0,
            policy=CustomActorCriticPolicy,
            policy_kwargs=dict(
                pi_net=RES_UNET,
                vf_net=CNN_FUNNEL,
                vf_latent_dim=16,
                pi_kwargs=dict(
                    sizes=[4, 8, 16, 16, 16]
                ),
                vf_kwargs=dict(
                    sizes=[4, 8, 16, 16, 16, 16, 16]
                ),
            ),
            n_steps=steps_per_rollout,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

        super().__init__(path, HeatInvaderEnv, env_kwargs, agent_kwargs, steps_per_rollout, n_envs)
