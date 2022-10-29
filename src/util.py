import numpy as np
from labellines import labelLines
from matplotlib import pyplot as plt
from stable_baselines3 import DDPG
from tqdm import tqdm

from src.agent.BaselineAgent import BaselineAgent
from src.agent.MPCAgent import MPCAgent
from src.agent.RandomAgent import RandomAgent


def plotGrid(listU, domain: int = None, dx: float = None, label: list[str] = None,
             xlim_min=0, xlim_max=None, ylim_min=-3.0, ylim_max=3.0,
             xlabel: str = "x(t)", ylabel: str = "u(t)",
             saveFig: str = None, linelabels: bool = False,
             render: bool = False):
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
                plt.plot(u, label=str(lab))
            else:
                plt.plot(x, u, label=str(lab))
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if linelabels:
        labelLines(plt.gca().get_lines(), zorder=2.5)
    elif label is not None:
        plt.legend()
    if saveFig is not None:
        plt.savefig(f'{saveFig}.pdf', bbox_inches='tight')
    if render:
        plt.show()

    plt.close()
    plt.cla()
    plt.clf()


def run_experiment(N: int = 1, _env=None, agent=None, ph: int = None,
                   save_model: str = None, load_model: str = None,
                   learn: bool = False, lr: float = 0.0001, n_epochs: int = 5,
                   render: bool = True, saveFig: str = None, linelabels: bool = False):
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
            pbar = tqdm(total=_env.step_count)
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
                if _env.step_idx in [5, 20, 99, 199, 399] and (render or (saveFig is not None)):
                    if render:
                        _env.enable_rendering()
                    if saveFig is not None:
                        ps = f't={_env.step_idx}'
                obs, reward, done, info = _env.step(action)
                rewards[i][_env.step_idx - 2] = reward
                if _env._render:
                    _env.disable_rendering()
                if saveFig is not None and _env.step_idx in [6, 21, 100, 200, 400]:
                    states_for_plot[ps] = obs
                pbar.update(1)
            final_states[i] = obs

    if render or (saveFig is not None):
        if agent is not None:
            # plot all the actions
            plotGrid(listU=actions, xlabel="t", ylabel="actions", xlim_max=_env.step_count,
                     ylim_min=np.min(actions), ylim_max=np.max(actions) + 0.1,
                     saveFig=f'{saveFig}_actions', render=render)
            if type(agent) == DDPG:
                # plot the rewards
                plotGrid(listU=rewards, xlabel="t", ylabel="rewards", xlim_max=_env.step_count,
                         ylim_min=np.min(rewards), ylim_max=np.max(rewards) + 0.1,
                         saveFig=f'{saveFig}_rewards', render=render)

        # plot the final state
        plot_values = states_for_plot.values()
        plot_labels = states_for_plot.keys()
        plotGrid(listU=plot_values, domain=_env.domain, dx=_env.dx,
                 label=plot_labels, saveFig=f'{saveFig}_states',
                 ylim_min=round(np.min(list(plot_values)), 2),
                 # ylim_min=min(round(np.min(list(plot_values)), 2), -1.0),
                 ylim_max=round(np.max(list(plot_values)), 2),
                 # ylim_max=max(round(np.max(list(plot_values)), 2), 1.0),
                 render=render, linelabels=linelabels)
    return np.array(final_states).mean(axis=0)
