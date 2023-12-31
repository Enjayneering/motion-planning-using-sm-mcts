import random
import numpy as np
import glob

from environment import *
from superparameter import *

# paths
path_to_repository = "/home/enjay/0_thesis/01_MCTS/"
path_to_data = path_to_repository+"data/"
path_to_src = path_to_repository+"src/"
path_to_tree = path_to_repository+"data/tree_{}.csv"
path_to_global_state = path_to_repository+"data/global_state.csv"
path_to_rollout_curr = path_to_repository+"data/rollout_curr.csv"
path_to_rollout_last = path_to_repository+"data/rollout_last.csv"
path_to_rollout_tmp = path_to_rollout_last + "~"
#path_to_rollout_longterm = path_to_repository+"data/rollout_longterm.csv"
path_to_results = path_to_repository+"results/"

# Video parameters
def get_next_video_name(path_to_results, environment_name=None):
    list_of_files = glob.glob(path_to_results + environment_name + "*.mp4")
    num_videos = len(list_of_files)
    next_video_name = "{}_{:02d}".format(environment_name, num_videos + 1)
    return next_video_name

next_video_name = get_next_video_name(path_to_results, environment_name=env.env_name_trigger[0][1])

# state and action space
state_space = ['x0', 'y0', 'theta0', 'x1', 'y1', 'theta1', 'timestep']
action_space = ['x0', 'y0', 'x1', 'y1']

# initialization parameters
#normalization parameters for UCB
max_payoff = 0
min_payoff = 0
payoff_range = max_payoff - min_payoff

#normalization parameters for payoff weights
aver_intermediate_penalties = 1
aver_final_payoff = 0



# motion parameters
delta_t = 1

freq_stat_data = 4
# Tuning parameters (weights for payoffs)

Competitive_params = {
    "action_set_0": {"velocity_0": np.linspace(0, 2, 3).tolist(),
                    "ang_velocity_0": np.linspace(-np.pi/2, np.pi/2, 3).tolist()},
    "action_set_1": {"velocity_1": np.linspace(0, 2, 3).tolist(),
                    "ang_velocity_1": np.linspace(-np.pi/2, np.pi/2, 3).tolist()},
    #"risk_factor_0": 0.1, #risk appetite agent1 = 1 - risk appetite agent0
}

MCTS_params = {
    "num_iter": 3000, #max number of simulations, proportional to complexity of the game

    "penalty_collision_0": 1, # exp scale
    "penalty_collision_1": 1,
    'penalty_collision_init': 0.1, # penalty at initial state
    'penalty_collision_delay': 1, # dynamic factor for ensuring reward is bigger than sum of penalties

    "reward_lead": 1,
    "reward_progress": 0,
    "penalty_centerline": -0,

    "penalty_stuck_in_env": -1,

    "c_param": np.sqrt(2), # c_param: exploration parameter | 3.52 - Tuned from Paper by Perick, 2012
    }


def is_terminal(state):
        # terminal condition
        if state.x0 >= env.dynamic_grid[state.timestep]['x_max']-1 or state.x1 >= env.dynamic_grid[state.timestep]['x_max']-1 or state.timestep >= game_horizon:
            return True
        return False

def generate_bernoulli(p):
    choices = [0, 1]
    probabilities = [1 - p, p]
    result = random.choices(choices, probabilities)[0]
    return result

def distance(state_0_xy, state_1_xy):
    x1 = state_0_xy[0]
    y1 = state_0_xy[1]
    x2 = state_1_xy[0]
    y2 = state_1_xy[1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)