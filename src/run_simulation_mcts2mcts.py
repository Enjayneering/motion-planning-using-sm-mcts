import os
import json
import time

from solvers.mcts_interface import run_mcts_interface
from folder_structure import *
from utilities.common_utilities import is_terminal
from utilities.plot_utilities import plot_single_run
from utilities.environment_utilities import Environment



# Import experiment configuration
from global_configs import mcts_vs_mcts


def run_simulation_mcts2mcts(sim_name, experiment, num_simulations):
    sim_data_dir = os.path.join(data_dir, sim_name)

    # Experiment within environment
    experiments = experiment.build_experiments()

    # create folder structure
    exp_raw_dir = os.path.join(sim_data_dir, "raw")
    exp_processed_dir = os.path.join(sim_data_dir, "processed")
    os.makedirs(exp_raw_dir, exist_ok=True)
    os.makedirs(exp_processed_dir, exist_ok=True)

    for experiment in experiments:

        # run multiple runs for statistical analysis
        for n in range(num_simulations):
            exp_data_dir = os.path.join(exp_raw_dir, experiment['name'], str(n))
            # Create the directory if it doesn't exist
            os.makedirs(exp_data_dir, exist_ok=True)

            conf_dict = experiment['dict']
            env_conf = experiment['dict']['env_conf']
            agent_conf = experiment['dict']['agent_conf']
            model_conf = experiment['dict']['model_conf']

            algo_conf_0 = experiment['dict']['algo_conf_0']
            algo_conf_1 = experiment['dict']['algo_conf_1']

            env=Environment(env_conf)


            curr_timestep = 0
            curr_joint_state = env.init_state+[curr_timestep] # list [x0, y0, theta0, x1, y1, theta1, timestep]

            # store results
            result_dict = {}
            result_dict['algo_data_0'] = []
            result_dict['algo_data_1'] = []
            trajectory_0 = [curr_joint_state[0:3]+[curr_timestep]]
            trajectory_1 = [curr_joint_state[3:6]+[curr_timestep]]

            start_time = time.time()
            while not is_terminal(env=env, state=curr_joint_state, timestep=curr_timestep, max_timestep=10):
                # run mcts for agent 0
                next_state_0, algo_data_0 = run_mcts_interface(ix_agent=0, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf, model_conf=model_conf, mcts_conf=algo_conf_0)
                
                # run mcts for agent 1
                next_state_1, algo_data_1 = run_mcts_interface(ix_agent=1, curr_state=curr_joint_state, env_conf=env_conf, agent_conf=agent_conf, model_conf=model_conf, mcts_conf=algo_conf_1)

                next_timestep = curr_timestep + 1
                trajectory_0.append(next_state_0+[next_timestep])
                trajectory_1.append(next_state_1+[next_timestep])
                # update current joint state
                curr_joint_state = next_state_0 + next_state_1 + [next_timestep]
                curr_timestep = next_timestep
                print(f"curr_joint_state: {curr_joint_state}")
                print(f"curr_timestep: {curr_timestep}")
                result_dict['algo_data_0'].append(algo_data_0)
                result_dict['algo_data_1'].append(algo_data_1)

            end_time = time.time()
            print
            # store results
            result_dict['trajectory_0'] = trajectory_0
            result_dict['trajectory_1'] = trajectory_1
            result_dict['exp_duration'] = float(end_time - start_time)
            result_dict['T_terminal'] = curr_timestep
            #result_dict['collision'] = 

            # Write the result dictionary to the JSON file
            with open(os.path.join(exp_data_dir, "conf_dict.json"), "w") as f:
                json.dump(conf_dict, f)
            with open(os.path.join(exp_data_dir, "result_dict.json"), "w") as f:
                json.dump(result_dict, f)
                
            # plot trajectory
            plot_single_run(env=env, path_savefig=exp_data_dir, result_dict=result_dict, main_agent=0)
            plot_single_run(env=env, path_savefig=exp_data_dir, result_dict=result_dict, main_agent=1)


if __name__ == '__main__':
    run_simulation_mcts2mcts(sim_name = "mcts_vs_mcts", experiment = mcts_vs_mcts, num_simulations = 10)