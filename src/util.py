import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DDPG

from src.agent.BaselineAgent import BaselineAgent
from src.agent.MPCAgent import MPCAgent
from src.agent.RandomAgent import RandomAgent


def plotGrid(listU, domain: int = None, dx: float = None, label: list[str] = None,
             xlim_min=0, xlim_max=None, ylim_min=-3.0, ylim_max=3.0,
             xlabel: str = "x(t)", ylabel: str = "u(t)",
             saveFig: str = None):
    x = None
    plt.tick_params(axis='x', which='minor', length=10)
    plt.grid(True, linestyle='--', which='both')

    if domain is not None and dx is not None:
        x = np.arange(0, domain, dx)
        if xlim_max is None:
            xlim_max = domain
    elif xlim_max is None:
        raise ValueError("param: xlim_max must be specified when domain is None")
    if label is None:
        for u in listU:
            if x is None:
                plt.plot(u)
            else:
                plt.plot(x, u)
    else:
        for u, lab in zip(listU, label):
            if x is None:
                plt.plot(u, label=lab)
            else:
                plt.plot(x, u, label=lab)
            plt.legend()
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if saveFig is not None:
        plt.savefig(f'{saveFig}.pdf', bbox_inches='tight')
    plt.show()


def run_experiment(N: int = 1, _env=None, agent=None, save_model: str = None, load_model: str = None,
                   learn: bool = False, lr: float = 0.0001,
                   n_epochs: int = 5, saveFig: str = None, render: bool = True, ph: int = None,
                   ymin_states=-0.5, ymax_states=2.0):
    if saveFig is not None and N > 1:
        raise ValueError("Currently saving figure is only implemented for single environment")

    states_for_plot = dict()
    final_states = np.zeros(N, dtype=np.ndarray)
    rewards = np.zeros((N, _env.step_count), dtype=float)
    actions = np.zeros((N, _env.step_count), dtype=float)

    if type(agent) == str:
        if agent == 'ddpg':
            if load_model is not None:
                agent = DDPG.load(load_model)
                agent.set_env(_env)
            else:
                agent = DDPG(verbose=0, env=_env, learning_rate=lr, policy='MlpPolicy')
        elif agent == 'mpc' or agent == 'baseline' or agent == 'random':
            if N > 1:
                raise Warning(f'using N = {N} for {agent} agent, will take a lot of time.')
            if learn:
                raise ValueError(f'{agent} can\'t learn, duh!!!')
            if save_model is not None:
                raise Warning(f'What model are you trying to save for {agent}? please invent an {agent} algorithm '
                              'that can learn first')
            if load_model is not None:
                raise Warning(f'load model for {agent}? Huh???')

            if agent == "mpc":
                agent = MPCAgent(env=_env) if ph is None else MPCAgent(env=_env, ph=ph)
            elif agent == "baseline":
                agent = BaselineAgent(env=_env, u_max=1.0, u_min=-1.0)
            else:
                agent = RandomAgent(env=_env)
    if learn:  # train
        agent = agent.learn(total_timesteps=_env.step_count * n_epochs)
        if save_model is not None:
            agent.save(save_model)
        return None
    else:  # test
        for i in range(N):
            obs = _env.reset()
            if saveFig is not None:
                states_for_plot["t=0"] = obs
            done = False
            ps = ""  # key value for states plot
            while not done:
                if agent is None:  # uncontrolled
                    action = [0.0]
                else:
                    action = agent.predict(observation=obs)
                    actions[i][_env.step_idx - 1] = action[0]
                if _env.step_idx in [1, 5, 20, 199] and (render or saveFig is not None):
                    if render:
                        _env.enable_rendering()
                    if saveFig is not None:
                        ps = f't={_env.step_idx}'
                obs, reward, done, info = _env.step(action)
                rewards[i][_env.step_idx - 2] = reward
                if _env._render:
                    _env.disable_rendering()
                if saveFig is not None and _env.step_idx in [2, 6, 21, 200]:
                    states_for_plot[ps] = obs
            final_states[i] = obs

    if render:
        # plot the final state
        plotGrid(listU=states_for_plot.values(), domain=_env.domain, dx=_env.dx,
                 label=states_for_plot.keys(), saveFig=f'{saveFig}_states',
                 ylim_min=ymin_states, ylim_max=ymax_states)

        # plot all the actions
        plotGrid(listU=actions, xlabel="t", ylabel="actions", xlim_max=_env.step_count,
                 ylim_min=np.min(actions), ylim_max=np.max(actions)+0.2,
                 saveFig=f'{saveFig}_actions')

        if type(agent) == DDPG:
            # plot the rewards
            plotGrid(listU=rewards, xlabel="t", ylabel="rewards", xlim_max=_env.step_count,
                     ylim_min=np.min(rewards), ylim_max=np.max(rewards) + 0.2,
                     saveFig=f'{saveFig}_rewards')

    return np.array(final_states).mean(axis=0)
