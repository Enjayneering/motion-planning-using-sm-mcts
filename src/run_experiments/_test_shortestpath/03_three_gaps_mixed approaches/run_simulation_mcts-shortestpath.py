import os
import json
import time
import pandas as pd
import multiprocessing

import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
exp_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
src_dir = os.path.abspath(os.path.join(exp_dir, os.pardir))
sys.path.insert(0, src_dir)
from solvers.bimatrix_interface_testplot import run_bimatrix_interface
from solvers.shortest_path_interface import run_shortestpath_interface
from solvers.mcts_interface import run_mcts_interface
from solvers.bimatrix_mcts.utilities.plot_test_utilities import *
from solvers.bimatrix_mcts.shortest_path_solver import ShortestPathSolver
from folder_structure import *
from utilities.common_utilities import is_terminal
from utilities.plot_utilities import plot_single_run
from utilities.config_utilities import Config
from utilities.environment_utilities import Environment



# Import experiment configuration
import exp_config


def run_simulation_bimatrix2bimatrix(sim_name, experiment, num_simulations):
    exp_raw_dir = os.path.join(data_dir, sim_name)

    # Experiment within environment
    experiments = experiment.build_experiments()

    # create folder structure
    os.makedirs(exp_raw_dir, exist_ok=True)

    for experiment in experiments:

        # run multiple runs for statistical analysis
        for n in range(num_simulations):
            exp_data_dir = os.path.join(exp_raw_dir, experiment['name'], str(n))
            # Create the directory if it doesn't exist
            os.makedirs(exp_data_dir, exist_ok=True)

            conf_dict = experiment['dict']
            env_conf = experiment['dict']['env_conf']
            model_conf = experiment['dict']['model_conf']

            agent_conf_0 = experiment['dict']['agent_conf_0']
            agent_conf_1 = experiment['dict']['agent_conf_1']

            algo_conf_0 = experiment['dict']['algo_conf_0']
            algo_conf_1 = experiment['dict']['algo_conf_1']

            game_dict =  {
                'name': env_conf['name'],
                'collision_distance': model_conf['collision_distance'],

                # MCTS Parameters
                'c_param': np.sqrt(2),
                'k_samples': 1, # not further investigated here
                'num_iter': algo_conf_0[0]['num_iter'],
                'gamma_exp3': algo_conf_0[0]['gamma_exp3'],

                # Engineering Parameters
                'alpha_rollout': 1,
                'alpha_terminal': algo_conf_0[0]['alpha_terminal'],
                'delta_t': model_conf['delta_t'],
                
                # Statistical Analysis -> NO
                'num_sim': 1,

                # Payoff Parameters
                'discount_factor': algo_conf_0[0]['discount_factor'],
                'weight_interm': algo_conf_0[0]['weight_interm'],
                'weight_final': algo_conf_0[0]['weight_final'],
                
                'weight_distance': algo_conf_0[0]['weight_distance'],
                'weight_collision': algo_conf_0[0]['weight_collision'],
                'weight_progress': algo_conf_0[0]['weight_progress'],
                'weight_lead': algo_conf_0[0]['weight_lead'],

                'weight_timestep': algo_conf_0[0]['weight_timestep'],
                'weight_winning': algo_conf_0[0]['weight_winning'], # better not, because too ambiguous
                'weight_final_lead': algo_conf_0[0]['weight_final_lead'],

                # Behavioural Parameters
                'collision_ignorance': 0.5, #[0,1] # Don't change
                
                'velocity_0': agent_conf_0[0]['velocity'],
                'ang_velocity_0': agent_conf_0[0]['ang_velocity'],
                'velocity_1': agent_conf_0[1]['velocity'],
                'ang_velocity_1': agent_conf_0[1]['ang_velocity'],

                'standard_dev_vel_0': np.max(agent_conf_0[0]['velocity']),
                'standard_dev_ang_vel_0': np.max(agent_conf_0[0]['ang_velocity']),
                'standard_dev_vel_1': np.max(agent_conf_0[1]['velocity']),
                'standard_dev_ang_vel_1': np.max(agent_conf_0[1]['ang_velocity']),

                'feature_flags': {
                    'run_mode': {'test': False, 'exp': True, 'live-plot': False},
                    'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
                    'collision_handling': {'punishing': True, 'pruning': False},
                    'selection_policy': algo_conf_0[0]['feature_flags']['selection_policy'],
                    'rollout_policy': {'random-uniform': False, 'random-informed': True},
                    'expansion_policy': algo_conf_0[0]['feature_flags']['expansion_policy'],
                    'strategy': algo_conf_0[0]['feature_flags']['strategy'],
                },
                'predef_traj': algo_conf_0[0]['predef_traj'],

            }

            env=Environment(env_conf)

            stop_event = multiprocessing.Event()  # Event to signal when plot.py should stop

            Game = ShortestPathSolver(env, Config(game_dict))
            # Create a process for the simulation
            simulation_process = multiprocessing.Process(target=run_simulation, args=(env, model_conf, exp_data_dir, agent_conf_0, agent_conf_1, algo_conf_0, algo_conf_1, env_conf, conf_dict))
            simulation_process.start()

            # Create a process for the test plotting
            #test_plotting_process = multiprocessing.Process(target=plot_independent, args=(Game, stop_event))
            #test_plotting_process.start()

            # Wait for the processes to finish
            simulation_process.join()
            #test_plotting_process.join()


def run_simulation(env, model_conf, exp_data_dir, agent_conf_0, agent_conf_1, algo_conf_0, algo_conf_1, env_conf, conf_dict):
    curr_timestep = 0
    curr_joint_state = env.init_state+[curr_timestep] # list [x0, y0, theta0, x1, y1, theta1, timestep]

    # store results
    result_dict = {}
    trajectory_0 = [curr_joint_state[0:3]+[curr_timestep]]
    trajectory_1 = [curr_joint_state[3:6]+[curr_timestep]]

    result_data_0 = []
    result_data_1 = []
    policy_data_0 = pd.DataFrame()
    policy_data_1 = pd.DataFrame()
    
    start_time = time.time()
    while not is_terminal(env=env, state=curr_joint_state, timestep=curr_timestep, max_timestep=10): # max timestep if algorithms go crazy
        print("Simulation timestep: \n", curr_timestep)
        
        
        # run algo for agent 0
        next_state_0, algo_data_0 = run_mcts_interface(ix_agent=0, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf_0, model_conf=model_conf, algo_conf=algo_conf_0)
        #next_state_0, algo_data_0 = run_bimatrix_interface(ix_agent=0, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf_0, model_conf=model_conf, algo_conf=algo_conf_0)
        #next_state_0, algo_data_0 = run_shortestpath_interface(ix_agent=0, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf_0, model_conf=model_conf, algo_conf=algo_conf_0)

        # run algo for agent 1
        #next_state_1, algo_data_1 = run_bimatrix_interface(ix_agent=1, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf_1, model_conf=model_conf, algo_conf=algo_conf_1)
        next_state_1, algo_data_1 = run_shortestpath_interface(ix_agent=1, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf_1, model_conf=model_conf, algo_conf=algo_conf_1)

        next_timestep = curr_timestep + model_conf['delta_t']

        trajectory_0.append(next_state_0+[next_timestep])
        trajectory_1.append(next_state_1+[next_timestep])

        # update current joint state
        curr_joint_state = next_state_0 + next_state_1 + [next_timestep]
        curr_timestep = next_timestep

        # store data

        # Add a new column 'timestep' to the DataFrame before concatenating
        try:
            algo_data_0["policy_df"]['timestep'] = curr_timestep
            result_data_0.append(algo_data_0["result_dict"])
            policy_data_0 = pd.concat([policy_data_0, algo_data_0["policy_df"]], axis=0)
        except:
            pass

        try:
            algo_data_1["policy_df"]['timestep'] = curr_timestep
            result_data_1.append(algo_data_1["result_dict"])
            policy_data_1 = pd.concat([policy_data_1, algo_data_1["policy_df"]], axis=0)
        except:
            pass
        
        """print(f"curr_joint_state: {curr_joint_state}")
        print(f"curr_timestep: {curr_timestep}")
        result_dict['algo_data_0'].append(algo_data_0)
        result_dict['algo_data_1'].append(algo_data_1)"""

    end_time = time.time()
    # store results
    result_dict['trajectory_0'] = trajectory_0
    result_dict['trajectory_1'] = trajectory_1
    result_dict['exp_duration'] = float(end_time - start_time)
    result_dict['T_terminal'] = curr_timestep

    # Write the result dictionary to the JSON file
    with open(os.path.join(exp_data_dir, "conf_dict.json"), "w") as f:
        json.dump(conf_dict, f)
    with open(os.path.join(exp_data_dir, "global_result_dict.json"), "w") as f:
        json.dump(result_dict, f)
    
    
    with open(os.path.join(exp_data_dir, "result_data_0.json"), "w") as f:
        json.dump(result_data_0, f)
    with open(os.path.join(exp_data_dir, "result_data_1.json"), "w") as f:
        json.dump(result_data_1, f)
    policy_data_0.to_csv(os.path.join(exp_data_dir, "policy_data_0.csv"))
    policy_data_1.to_csv(os.path.join(exp_data_dir, "policy_data_1.csv"))
        
    # plot trajectory
    plot_single_run(env=env, path_savefig=exp_data_dir, result_dict=result_dict, main_agent=0)
    plot_single_run(env=env, path_savefig=exp_data_dir, result_dict=result_dict, main_agent=1)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
    parent_dir_foldername = os.path.basename(parent_dir)

    run_simulation_bimatrix2bimatrix(sim_name = parent_dir_foldername, experiment = exp_config, num_simulations = 10)

    