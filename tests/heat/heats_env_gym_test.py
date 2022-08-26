import unittest
from typing import Optional

import gym
from phi.flow import *
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd

from src.env.heat_env_gym import HeatEnvGym
from src.networks import RES_UNET, CNN_FUNNEL
from src.policy import CustomActorCriticPolicy


class HeatEnvGymTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.N = 32
        cls.num_envs = 1
        cls.step_count = 32
        cls.domain = Domain((cls.N,), box=box[0:1])
        cls.dt = 1. / cls.step_count
        cls.default_diffusivity = 0.01 / (cls.N * np.pi)
        cls.final_reward_factor = 32
        cls.reward_rms: Optional[RunningMeanStd] = None
        cls.exp_name = 'v0'
        cls.data_path = 'forced-heat-clash'
        cls.agent_krargs = dict(verbose=0, policy=CustomActorCriticPolicy,
                                policy_kwargs=dict(pi_net=RES_UNET, vf_net=CNN_FUNNEL, vf_latent_dim=16,
                                                   pi_kwargs=dict(
                                                       sizes=[4, 8, 16, 16, 16]
                                                   ),
                                                   vf_kwargs=dict(
                                                       sizes=[4, 8, 16, 16, 16]
                                                   ), ),
                                n_steps=32,
                                n_epochs=100,
                                learning_rate=1e-4,
                                batch_size=32)

    def test_init_env(self):
        env = HeatEnvGym(N=self.N,
                         num_envs=self.num_envs,
                         domain=self.domain, dt=self.dt, default_diffusivity=self.default_diffusivity,
                         final_reward_factor=self.final_reward_factor, reward_rms=self.reward_rms,
                         exp_name=self.exp_name)

        assert isinstance(env, gym.Env)
        self.env = env

    def test_reset_step_env(self):
        # test initialisation
        global obs
        self.test_init_env()
        assert self.env is not None

        # initialize agent
        self.agent = PPO(env=self.env, **self.agent_krargs)
        assert self.agent is not None
        # train agent
        print("training begin")
        self.agent.learn(total_timesteps=32)
        print("training complete")

        # testing
        # done = False
        # while not done:
        #     obs = self.env.reset()
        #     act, _ = self.agent.predict(obs)
        #     obs, _, dones, _ = self.env.step(act)
        #     done = dones[0]

        # fig = plt.figure().gca()
        # fig = self.env.show_state([(obs, "prediction")], fig)
        # fig = self.env.show_state([(self.env.gt_state, "gt")], fig)
        # fig = self.env.show_state([(self.env.goal_state, "goalstate")], fig)
        # plt.xlabel('x')
        # plt.ylabel('u')
        # plt.legend()
        # plt.title("Prediction and gt")
        # plt.show(warn=False)

        # obs = self.env.reset()
        # assert isinstance(obs, np.ndarray)
        # self.force = GaussianClash(self.num_envs)
        #
        # fig = plt.figure().gca()
        # for i in range(5):
        #     # Train PPO
        #
        #     idx = np.random.uniform(0, 1, size=self.step_count)
        #     action = self.force.sample_at(idx)
        #     obs, rw, done, info = self.env.step(action)
        #     # obs.shape = (1, self.step_count, 3) where 3 = (cont_state, goal_state, time_info)
        #     # So, resizing obs by extracting cont_state for plotting
        #     cont_state = obs.reshape((self.step_count, len(obs.shape)))[:, 0]
        #     fig = self.env.show_state([(self.env.goal_state, "goal_state"),
        #                                (self.env.gt_state, "gt_state"),
        #                                (cont_state, "cont_state")], fig=fig)
        #
        # plt.xlabel('x')
        # plt.ylabel('u')
        # plt.legend()
        # plt.title("goal, gt, cont states")
        # plt.show(warn=False)


if __name__ == '__main__':
    unittest.main()
