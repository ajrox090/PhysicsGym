from typing import Optional
from phi.flow import *
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.ppo import MlpPolicy

from src.env.heat_env_gym import Heat1DEnvGym
from src.runner import RLRunner

runner = RLRunner(path_config=".../experiment.yml")
rc_env = runner.config['env']
rc_agent = runner.config['agent']
# env
N = rc_env['N']
step_count = rc_env['step_count']
domain_dict = dict(x=N, bounds=Box[0:1])
dt = 1. / step_count
diffusivity = 0.01 / (N * np.pi)
if 'diffusivity' in rc_env.keys():
    diffusivity = rc_env['diffusivity']
final_reward_factor = rc_env['final_reward_factor']
reward_rms: Optional[RunningMeanStd] = None

# agent
num_epochs = rc_agent['num_epochs']
lr = rc_agent['lr']
batch_size = step_count
env_krargs = dict(N=N,
                  domain_dict=domain_dict, dt=dt, step_count=step_count,
                  diffusivity=diffusivity,
                  final_reward_factor=final_reward_factor,
                  reward_rms=reward_rms)
agent_krargs = dict(verbose=0, policy=MlpPolicy,
                    n_steps=step_count,
                    n_epochs=num_epochs,
                    learning_rate=lr,
                    batch_size=batch_size)

env = Heat1DEnvGym(**env_krargs)
agent = PPO(env=env, **agent_krargs)

print("training begins")
env.enable_rendering()
agent.learn(total_timesteps=32)
print("training complete")

obs = env.reset()
obs2 = obs[:, :1]
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
# vis.plot(curr_state)
# vis.show()
curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)

# 4.3) Play :D
# The view below is a very nice interactive viewer by phiflow, this basically plots every phi.Field objects in an
# interactive plot which opens in a browser. The objects for plotting can also be described as parameters.
# For the below example, the supported object is curr_state.
for i in view(play=True, namespace=globals()).range(32):
    curr_state = CenteredGrid(phi.math.tensor(obs2, env.cont_state.shape), obs2.shape)
    act = agent.predict(obs)[0]
    obs, reward, done, info = env.step(act)


class HeatEnvGymTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.N = 32
        cls.step_count = 32
        cls.domain = Domain((cls.N,), box=Box[0:1])
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
        env = Heat1DEnvGym(N=self.N,
                           domain=self.domain,
                           step_count=self.step_count,
                           dt=self.dt,
                           default_diffusivity=self.default_diffusivity,
                           final_reward_factor=self.final_reward_factor,
                           reward_rms=self.reward_rms,
                           exp_name=self.exp_name)

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
