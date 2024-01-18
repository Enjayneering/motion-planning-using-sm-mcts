import random
import numpy as np

# paths
path_to_repository = "/home/enjay/0_thesis/01_MCTS/"
path_to_src = path_to_repository+"src/"

# test file structure
path_to_tests = path_to_repository+"tests/"
path_to_results = path_to_tests+"results/"
path_to_data = path_to_tests+"data/"
path_to_trees = path_to_data+"trees/"
path_to_tree = path_to_trees+"/tree_{}.csv"
path_to_global_state = path_to_data+"global_state.csv"
path_to_rollout_curr = path_to_data+"rollout_curr.csv"
path_to_rollout_last = path_to_data+"rollout_last.csv"
path_to_rollout_tmp = path_to_rollout_last + "~"


# experimental file structure
path_to_experiments = path_to_repository+"experiments/"
new_exp_folder = path_to_experiments+""

#normalization parameters for UCB
max_payoff = 0
min_payoff = 0
payoff_range = max_payoff - min_payoff

#normalization parameters for payoff weights
aver_intermediate_penalties = 1
aver_final_payoff = 0

freq_stat_data = 2

def get_max_timehorizon(config):
    return config.alpha_t * config.terminal_progress

def is_terminal(Game, state):
        # terminal condition
        if state.x0 >= Game.config.terminal_progress or state.x1 >= Game.config.terminal_progress:
            print("Terminal state reached")
            return True
        elif state.timestep >= get_max_timehorizon(Game.config):
            print("Max timehorizon reached")
            return True
        else:
            return False

"""def generate_bernoulli(p):
    choices = [0, 1]
    probabilities = [1 - p, p]
    result = random.choices(choices, probabilities)[0]
    return result"""

def get_winner(state):
    if state.x0 > state.x1: #agent 0 is ahead
        return [1,0]
    elif state.x0 < state.x1: #agent 1 is ahead
        return [0,1]
    else: #draw
        return [0,0]