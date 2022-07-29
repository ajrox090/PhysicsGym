# import tensorforce
# from dolfin import *
# from tqdm import tqdm
# from msh_convert import convert
# from flow_solver import FlowSolver
# from generate_msh import generate_mesh
# from dolfin import Expression, File, plot
# from tensorforce.environments import Environment
# from printind.printind_function import printi, printiv
# from printind.printind_decorators import printi_all_method_calls as printidc
# from probes import PenetratedDragProbeANN, PenetratedLiftProbeANN, PressureProbeANN, VelocityProbeANN, \
#     RecirculationAreaProbe
import os
import csv
import sys
import pickle

import gym
import numpy as np
import random as random

cwd = os.getcwd()
sys.path.append(cwd + "/../Simulation/")

"""
# TODO: check that right types etc from tensorfoce examples
# typically:

# from tensorforce.contrib.openai_gym import OpenAIGym
# environment = OpenAIGym('MountainCarContinuous-v0', visualize=False)

# printiv(environment.states)
# environment.states = {'shape': (2,), 'type': 'float'}
# printiv(environment.actions)
# environment.actions = {'max_value': 1.0, 'shape': (1,), 'min_value': -1.0, 'type': 'float'}
"""


def constant_profile(mesh, degree):
    '''
    Time independent inflow profile.
    '''
    bot = mesh.coordinates().min(axis=0)[1]
    top = mesh.coordinates().max(axis=0)[1]

    H = top - bot

    Um = 1.5

    return Expression(('-4*Um*(x[1]-bot)*(x[1]-top)/H/H',
                       '0'), bot=bot, top=top, H=H, Um=Um, degree=degree, time=0)


class RingBuffer:
    "A 1D ring buffer using numpy arrays"

    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) % self.data.size
        return self.data[idx]


class Env2DCylinder(gym.Env):
    """Environment for 2D flow simulation around a cylinder."""

    def __init__(self, path_root, geometry_params, flow_params, solver_params, output_params, optimization_params,
                 inspection_params, n_iter_make_ready=None, verbose=0, size_history=2000, reward_function='plain_drag',
                 size_time_state=50, number_steps_execution=1, simu_name="Simu"):

        super().__init__()
        self.p_ = None
        self.u_ = None
        self.area_probe = None
        self.u_out = None
        self.recirc_area = None
        self.lift = None
        self.drag = None
        self.ann_probes = None
        self.Qs = None
        self.lift_probe = None
        self.drag_probe = None
        self.flow = None
        self.ready_to_use = None
        self.probes_values = None
        self.history_parameters = None
        self.initialized_output = None
        self.accumulated_lift = None
        self.accumulated_drag = None
        self.solver_step = None
        print("--- call init ---")

        self.path_root = path_root
        self.flow_params = flow_params
        self.geometry_params = geometry_params
        self.solver_params = solver_params
        self.output_params = output_params
        self.optimization_params = optimization_params
        self.inspection_params = inspection_params
        self.verbose = verbose
        self.n_iter_make_ready = n_iter_make_ready
        self.size_history = size_history
        self.reward_function = reward_function
        self.size_time_state = size_time_state
        self.number_steps_execution = number_steps_execution

        self.simu_name = simu_name

        self.last_episode_number = 0
        self.episode_number = 0
        self.episode_drags = np.array([])
        self.episode_areas = np.array([])
        self.episode_lifts = np.array([])

        self.initialized_visualization = False

        self.start_class(complete_reset=True)

        print("--- done init ---")

    def start_class(self, complete_reset=True):
        self.solver_step = 0
        self.accumulated_drag = 0
        self.accumulated_lift = 0

        self.initialized_output = False

        self.resetted_number_probes = False

        self.area_probe = None

        self.history_parameters = {}

        for crrt_jet in range(len(self.geometry_params["jet_positions"])):
            self.history_parameters["jet_{}".format(crrt_jet)] = RingBuffer(self.size_history)

        self.history_parameters["number_of_jets"] = len(self.geometry_params["jet_positions"])

        for crrt_probe in range(len(self.output_params["locations"])):
            if self.output_params["probe_type"] == 'pressure':
                self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(self.size_history)
            elif self.output_params["probe_type"] == 'velocity':
                self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(self.size_history)
                self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(self.size_history)

        self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

        self.history_parameters["drag"] = RingBuffer(self.size_history)
        self.history_parameters["lift"] = RingBuffer(self.size_history)
        self.history_parameters["recirc_area"] = RingBuffer(self.size_history)

        # ------------------------------------------------------------------------
        # remesh if necessary
        h5_file = '.'.join([self.path_root, 'h5'])
        msh_file = '.'.join([self.path_root, 'msh'])
        self.geometry_params['mesh'] = h5_file

        # Regenerate mesh?
        if self.geometry_params['remesh']:

            if self.verbose > 0:
                printi("Remesh")
                printi("generate_mesh start...")

            generate_mesh(self.geometry_params, template=self.geometry_params['template'])

            if self.verbose > 0:
                printi("generate_mesh done!")

            assert os.path.exists(msh_file)

            convert(msh_file, h5_file)
            assert os.path.exists(h5_file)

        # ------------------------------------------------------------------------
        # if necessary, load initialization fields
        if self.n_iter_make_ready is None:
            if self.verbose > 0:
                printi("Load initial flow")

            self.flow_params['u_init'] = 'mesh/u_init.xdmf'
            self.flow_params['p_init'] = 'mesh/p_init.xdmf'

            if self.verbose > 0:
                printi("Load buffer history")

            with open('mesh/dict_history_parameters.pkl', 'rb') as f:
                self.history_parameters = pickle.load(f)

            if not "number_of_probes" in self.history_parameters:
                self.history_parameters["number_of_probes"] = 0

            if not "number_of_jets" in self.history_parameters:
                self.history_parameters["number_of_jets"] = len(self.geometry_params["jet_positions"])
                printi("Warning!! The number of jets was not set in the loaded hdf5 file")

            if not "lift" in self.history_parameters:
                self.history_parameters["lift"] = RingBuffer(self.size_history)
                printi("Warning!! No value for the lift founded")

            if not "recirc_area" in self.history_parameters:
                self.history_parameters["recirc_area"] = RingBuffer(self.size_history)
                printi("Warning!! No value for the recirculation area founded")

            # if not the same number of probes, reset
            if not self.history_parameters["number_of_probes"] == len(self.output_params["locations"]):
                for crrt_probe in range(len(self.output_params["locations"])):
                    if self.output_params["probe_type"] == 'pressure':
                        self.history_parameters["probe_{}".format(crrt_probe)] = RingBuffer(self.size_history)
                    elif self.output_params["probe_type"] == 'velocity':
                        self.history_parameters["probe_{}_u".format(crrt_probe)] = RingBuffer(self.size_history)
                        self.history_parameters["probe_{}_v".format(crrt_probe)] = RingBuffer(self.size_history)

                self.history_parameters["number_of_probes"] = len(self.output_params["locations"])

                printi("Warning!! Number of probes was changed! Probes buffer content reseted")

                self.resetted_number_probes = True

        # ------------------------------------------------------------------------
        # create the flow simulation object
        self.flow = FlowSolver(self.flow_params, self.geometry_params, self.solver_params)

        # ------------------------------------------------------------------------
        # Setup probes
        if self.output_params["probe_type"] == 'pressure':
            self.ann_probes = PressureProbeANN(self.flow, self.output_params['locations'])

        elif self.output_params["probe_type"] == 'velocity':
            self.ann_probes = VelocityProbeANN(self.flow, self.output_params['locations'])
        else:
            raise RuntimeError("unknown probe type")

        # Setup drag measurement
        self.drag_probe = PenetratedDragProbeANN(self.flow)
        self.lift_probe = PenetratedLiftProbeANN(self.flow)

        # ------------------------------------------------------------------------
        # No flux from jets for starting
        self.Qs = np.zeros(len(self.geometry_params['jet_positions']))

        # ------------------------------------------------------------------------
        # prepare the arrays for plotting positions
        self.compute_positions_for_plotting()

        # ------------------------------------------------------------------------
        # if necessary, make converge
        if self.n_iter_make_ready is not None:
            self.u_, self.p_ = self.flow.evolve(self.Qs)
            path = ''
            if "dump" in self.inspection_params:
                path = 'results/area_out.pvd'
            self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)
            if self.verbose > 0:
                printi("Compute initial flow")
                printiv(self.n_iter_make_ready)

            for _ in range(self.n_iter_make_ready):
                self.u_, self.p_ = self.flow.evolve(self.Qs)

                self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
                self.drag = self.drag_probe.sample(self.u_, self.p_)
                self.lift = self.lift_probe.sample(self.u_, self.p_)
                self.recirc_area = self.area_probe.sample(self.u_, self.p_)

                self.write_history_parameters()
                self.visual_inspection()
                self.output_data()

                self.solver_step += 1

        if self.n_iter_make_ready is not None:
            encoding = XDMFFile.Encoding_HDF5
            mesh = convert(msh_file, h5_file)
            comm = mesh.mpi_comm()

            # save field data
            XDMFFile(comm, 'mesh/u_init.xdmf').write_checkpoint(self.u_, 'u0', 0, encoding)
            XDMFFile(comm, 'mesh/p_init.xdmf').write_checkpoint(self.p_, 'p0', 0, encoding)

            # save buffer dict
            with open('mesh/dict_history_parameters.pkl', 'wb') as f:
                pickle.dump(self.history_parameters, f, pickle.HIGHEST_PROTOCOL)

        # ----------------------------------------------------------------------
        # if reading from disk, show to check everything ok
        if self.n_iter_make_ready is None:
            # Let's start in a random position of the vortex shading
            if self.optimization_params["random_start"]:
                rd_advancement = np.random.randint(650)
                for j in range(rd_advancement):
                    self.flow.evolve(self.Qs)
                print("Simulated {} iterations before starting the control".format(rd_advancement))

            self.u_, self.p_ = self.flow.evolve(self.Qs)
            path = ''
            if "dump" in self.inspection_params:
                path = 'results/area_out.pvd'
            self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)

            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.drag_probe.sample(self.u_, self.p_)
            self.lift = self.lift_probe.sample(self.u_, self.p_)
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            self.write_history_parameters()
            # self.visual_inspection()
            # self.output_data()

            # self.solver_step += 1

            # time.sleep(10)

        # ----------------------------------------------------------------------
        # if necessary, fill the probes buffer
        if self.resetted_number_probes:
            printi("Need to fill again the buffer; modified number of probes")

            for _ in range(self.size_history):
                self.execute()

        # ----------------------------------------------------------------------
        # ready now

        # Initialisation du prob de recirculation area
        # path=''
        # if "dump" in self.inspection_params:
        #    path = 'results/area_out.pvd'
        # self.area_probe = RecirculationAreaProbe(self.u_, 0, store_path=path)

        self.ready_to_use = True

    def __str__(self):
        printi("Env2DCylinder ---")

    def close(self):
        """
        Close environment. No other method calls possible afterwards.
        """

        self.ready_to_use = False

    def seed(self, seed):
        """
        Sets the random seed of the environment to the given value (current time, if seed=None).
        Naturally deterministic Environments don't have to implement this method.

        Args:
            seed (int): The seed to use for initializing the pseudo-random number generator (default=epoch time in sec).
        Returns: The actual seed (int) used OR None if Environment did not override this method (no seeding supported).
        """

        # we have a deterministic environment: no need to implement

        return None

    def reset(self):
        """
        Reset environment and setup for new episode.

        Returns:
            initial state of reset environment.
        """

        if self.solver_step > 0:
            mean_accumulated_drag = self.accumulated_drag / self.solver_step
            mean_accumulated_lift = self.accumulated_lift / self.solver_step

            if self.verbose > -1:
                printi("mean accumulated drag on the whole episode: {}".format(mean_accumulated_drag))

        if self.inspection_params["show_all_at_reset"]:
            self.show_drag()
            self.show_control()

        chance = random.random()

        probability_hard_reset = 0.2

        if chance < probability_hard_reset:
            self.start_class(complete_reset=True)
        else:
            self.start_class(complete_reset=False)
        # print(chance)

        next_state = np.transpose(np.array(self.probes_values))
        if self.verbose > 0:
            printiv(next_state)

        self.episode_number += 1

        return (next_state)

    def execute(self, actions=None):

        if self.verbose > 1:
            printi("--- call execute ---")

        if actions is None:
            if self.verbose > -1:
                printi("carefull, no action given; by default, no jet!")

            nbr_jets = len(self.geometry_params["jet_positions"])
            actions = np.zeros((nbr_jets,))

        if self.verbose > 2:
            printiv(actions)

        # to execute several numerical integration steps
        for _ in range(self.number_steps_execution):

            # try to force a continuous / smoothe(r) control
            if "smooth_control" in self.optimization_params:
                # printiv(self.optimization_params["smooth_control"])
                # printiv(actions)
                # printiv(self.Qs)
                self.Qs += self.optimization_params["smooth_control"] * (np.array(actions) - self.Qs)
            else:
                self.Qs = np.transpose(np.array(actions))

            # impose a zero net Qs
            if "zero_net_Qs" in self.optimization_params:
                if self.optimization_params["zero_net_Qs"]:
                    self.Qs = self.Qs - np.mean(self.Qs)

            # evolve one numerical timestep forward
            self.u_, self.p_ = self.flow.evolve(self.Qs)

            # displaying information that has to do with the solver itself
            self.visual_inspection()
            self.output_data()

            # we have done one solver step
            self.solver_step += 1

            # sample probes and drag
            self.probes_values = self.ann_probes.sample(self.u_, self.p_).flatten()
            self.drag = self.drag_probe.sample(self.u_, self.p_)
            self.lift = self.lift_probe.sample(self.u_, self.p_)
            self.recirc_area = self.area_probe.sample(self.u_, self.p_)

            # write to the history buffers
            self.write_history_parameters()

            self.accumulated_drag += self.drag
            self.accumulated_lift += self.lift

        # TODO: the next_state may incorporate more information: maybe some time information?
        next_state = np.transpose(np.array(self.probes_values))

        if self.verbose > 2:
            printiv(next_state)

        terminal = False

        if self.verbose > 2:
            printiv(terminal)

        reward = self.compute_reward()

        if self.verbose > 2:
            printiv(reward)

        if self.verbose > 1:
            printi("--- done execute ---")

        return next_state, terminal, reward

    def compute_reward(self):
        # NOTE: reward should be computed over the whole number of iterations in each execute loop
        if self.reward_function == 'plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"].get()[-self.number_steps_execution:]
            return (np.mean(
                values_drag_in_last_execute) + 0.159)  # TODO: the 0.159 value is a proxy value corresponding to the
            # mean drag when no control; may depend on the geometry
        elif self.reward_function == 'recirculation_area':
            return - self.area_probe.sample(self.u_, self.p_)
        elif self.reward_function == 'max_recirculation_area':
            return self.area_probe.sample(self.u_, self.p_)
        elif self.reward_function == 'drag':  # a bit dangerous, may be injecting some momentum
            return self.history_parameters["drag"].get()[-1] + 0.159
        elif self.reward_function == 'drag_plain_lift':  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.number_steps_execution)
            avg_drag = np.mean(self.history_parameters["drag"].get()[-avg_length:])
            avg_lift = np.mean(self.history_parameters["lift"].get()[-avg_length:])
            return avg_drag + 0.159 - 0.2 * abs(avg_lift)
        elif self.reward_function == 'max_plain_drag':  # a bit dangerous, may be injecting some momentum
            values_drag_in_last_execute = self.history_parameters["drag"].get()[-self.number_steps_execution:]
            return - (np.mean(values_drag_in_last_execute) + 0.159)
        elif self.reward_function == 'drag_avg_abs_lift':  # a bit dangerous, may be injecting some momentum
            avg_length = min(500, self.number_steps_execution)
            avg_abs_lift = np.mean(np.absolute(self.history_parameters["lift"].get()[-avg_length:]))
            avg_drag = np.mean(self.history_parameters["drag"].get()[-avg_length:])
            return avg_drag + 0.159 - 0.2 * avg_abs_lift

        # TODO: implement some reward functions that take into account how much energy / momentum we inject into the
        #  flow

        else:
            raise RuntimeError("reward function {} not yet implemented".format(self.reward_function))

    @property
    def states(self):
        """
        Return the state space. Might include subdicts if multiple states are available simultaneously.

        Returns: dict of state properties (shape and type).

        """

        if self.output_params["probe_type"] == 'pressure':
            return dict(type='float',
                        shape=(len(self.output_params["locations"]) * self.optimization_params[
                            "num_steps_in_pressure_history"],)
                        )

        elif self.output_params["probe_type"] == 'velocity':
            return dict(type='float',
                        shape=(2 * len(self.output_params["locations"]) * self.optimization_params[
                            "num_steps_in_pressure_history"],)
                        )

    @property
    def actions(self):
        """
        Return the action space. Might include subdicts if multiple actions are available simultaneously.

        Returns: dict of action properties (continuous, number of actions).
        """

        # NOTE: we could also have several levels of dict in dict, for example:
        # return { str(i): dict(continuous=True, min_value=0, max_value=1) for i in range(self.n + 1) }

        return dict(type='float',
                    shape=(len(self.geometry_params["jet_positions"]),),
                    min_value=self.optimization_params["min_value_jet_MFR"],
                    max_value=self.optimization_params["max_value_jet_MFR"])
