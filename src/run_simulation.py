import os
import sys
"""
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the child utilities directory to sys.path
sys.path.append(current_dir+"solvers")
sys.path.append(current_dir+"solvers/sm_mcts")
sys.path.append(current_dir+"solvers/sm_mcts/utilities")"""

from solvers import *
from experiment_configs import *
from utilities import *

from solvers.mcts_interface import run_mcts_interface
from experiment_configs.env_conf import env_conf
from experiment_configs.model_conf import model_conf
from experiment_configs.agent_conf import agent_conf
from experiment_configs.mcts_conf import mcts_conf
from folder_structure import *
from utilities.common_utilities import is_terminal
from utilities.plot_utilities import plot_single_run
from utilities.environment import Environment

import json
import time

if __name__ == '__main__':
    # Experiment within environment
    ExpEnv = Environment(env_conf)

    curr_timestep = 0
    curr_joint_state = ExpEnv.init_state+[curr_timestep] # list [x0, y0, theta0, x1, y1, theta1, timestep]

    # store results
    result_dict = {}
    trajectory_0 = []
    trajectory_1 = []

    start_time = time.time()
    while not is_terminal(env=ExpEnv, state=curr_joint_state, timestep=curr_timestep, max_timestep=12):
        # run mcts for agent 0
        next_joint_state_0, algo_data_0 = run_mcts_interface(ix_agent=0, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf, model_conf=model_conf, mcts_conf=mcts_conf)
        next_state_0 = next_joint_state_0[0:3]
        trajectory_0.append(next_state_0+[curr_timestep])
        # run mcts for agent 1
        next_joint_state_1, algo_data_1 = run_mcts_interface(ix_agent=1, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf, model_conf=model_conf, mcts_conf=mcts_conf)
        next_state_1 = next_joint_state_1[3:]
        trajectory_1.append(next_state_1+[curr_timestep])
        # update current joint state
        curr_timestep += 1
        curr_joint_state = next_state_0 + next_state_1 + [curr_timestep]
        print(f"curr_joint_state: {curr_joint_state}")
        print(f"curr_timestep: {curr_timestep}")

    end_time = time.time()
    print
    # store results
    result_dict['trajectory_0'] = trajectory_0
    result_dict['trajectory_1'] = trajectory_1
    result_dict['T_terminal'] = curr_timestep
    result_dict['exp_duration'] = end_time - start_time
    #result_dict['collision'] = 

    # Write the result dictionary to the JSON file
    with open(os.path.join(results_dir, "algo_data_0.json"), "w") as f:
        json.dump(algo_data_0, f)
    with open(os.path.join(results_dir, "algo_data_1.json"), "w") as f:
        json.dump(algo_data_1, f)
        
    # plot trajectory
    plot_single_run(env=ExpEnv, path_savefig=results_dir, result_dict=result_dict, main_agent=0)
    plot_single_run(env=ExpEnv, path_savefig=results_dir, result_dict=result_dict, main_agent=1)
