import os
import time
import pandas as pd

import matplotlib.pyplot as plt

from .bimatrix_mcts_objects import State, run_mcts
from .utilities.common_utilities import *
from .utilities.kinodynamic_utilities import *
from .utilities.payoff_utilities import *
from .utilities.csv_utilities import *
from .utilities.networkx_utilities import *
from .utilities.config_utilities import *
from .utilities.environment_utilities import *

from .shortest_path_planner import ShortestPathPlanner


class ShortestPathSolver:
    def __init__(self, env, game_config, init_state=None):        
        self.config = game_config
        self.env = env
        if init_state is not None:
            # init state is list
            self.init_state = init_state
        else:       
            self.init_state = [self.env.init_state[0], self.env.init_state[1], self.env.init_state[2], self.env.init_state[3], self.env.init_state[4], self.env.init_state[5]]
            #[self.env.init_state['x0'], self.env.init_state['y0'], self.env.init_state['theta0'], self.env.init_state['x1'], self.env.init_state['y1'], self.env.init_state['theta1']]
        self.terminal_state = [self.env.goal_state['x0'], self.env.goal_state['y0'], self.env.goal_state['theta0'], self.env.goal_state['x1'], self.env.goal_state['y1'], self.env.goal_state['theta1']]
        self.name = get_next_game_name(path_to_results, game_config.name)

        self.Model_params = self._set_model_params()
        self.Kinodynamic_params = self._set_kinodynamic_params()

        self.config['max_timehorizon'] = get_max_timehorizon(self)

    def _set_model_params(self):
        Model_params = {
            "state_space": ['x0', 'y0', 'theta0', 'x1', 'y1', 'theta1', 'timestep'],
            "delta_t": self.config.delta_t,
            "collision_distance": self.config.collision_distance,
            "agents": [0, 1],
            }
        return Model_params
    

    def _get_goal_state(self):
        goal_state = State(x0=self.env.goal_state['x0'],
                            y0=self.env.goal_state['y0'],
                            theta0=self.env.goal_state['theta0'],
                            x1=self.env.goal_state['x1'],
                            y1=self.env.goal_state['y1'],
                            theta1=self.env.goal_state['theta1'],
                            timestep=get_max_timehorizon(self))
        return goal_state
    
    def _set_kinodynamic_params(self):
        Kinodynamic_params = {
        "action_set_0": {"velocity": self.config.velocity_0,
                        "ang_velocity": self.config.ang_velocity_0},
        "action_set_1": {"velocity": self.config.velocity_1,
                        "ang_velocity": self.config.ang_velocity_1},
        }
        return Kinodynamic_params

    def _init_state(self):
        # initialize root node
        initial_state = State(x0=self.init_state[0],
                            y0=self.init_state[1],
                            theta0=self.init_state[2],
                            x1=self.init_state[3],
                            y1=self.init_state[4],
                            theta1=self.init_state[5],
                            timestep=0)
        return initial_state
    
    
    def get_next_action(self, agent):
        # timestep sim, so that we can use the algorithm for simulation (only run for one stage the selfplay)

        # create a text trap and redirect stdout
        #text_trap = io.StringIO()
        #sys.stdout = text_trap

        # RUN A FULL SIMULATION OF A COMPETITIVE RACING GAME

        current_state = self.init_state

        if agent == 0:
            curr_state_own = current_state[:3]
            curr_state_opp = current_state[3:6]
        else:
            curr_state_own = current_state[3:6]
            curr_state_opp = current_state[:3]
        SPPlanner = ShortestPathPlanner(self.env, action_set=self.Kinodynamic_params['action_set_{}'.format(agent)], curr_state=curr_state_own, opponent_state=curr_state_opp, start_timestep=current_state[6], ix_agent=agent, dt=self.Model_params["delta_t"])
        traj_dict = SPPlanner.get_trajectories(num_trajectories=1)
        action_trajectory = traj_dict['actions']
        state_trajectory = traj_dict['states']
       
        return action_trajectory, state_trajectory
    