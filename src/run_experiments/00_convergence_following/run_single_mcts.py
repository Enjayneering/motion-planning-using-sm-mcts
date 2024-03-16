import time
import multiprocessing
import json
import itertools
import sys

import exp_config

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
src_dir = os.path.abspath(os.path.join(exp_dir, os.pardir))
sys.path.insert(0, src_dir)

from utilities.config_utilities import *
from folder_structure import *
from solvers.sm_mcts.competitive_game import CompetitiveGame
from solvers.sm_mcts.utilities.plot_test_utilities import *
from solvers.sm_mcts.utilities.plot_utilities import *
from solvers.sm_mcts.utilities.common_utilities import *
from solvers.sm_mcts.utilities.csv_utilities import *

from solvers.sm_mcts.utilities.run_utilities import *
from solvers.sm_mcts.utilities.environment_utilities import *



def run_test(game_dict):
    GamePlanner = CompetitiveGame(Environment(Config(game_dict)), Config(game_dict))
    for _ in range(GamePlanner.config.num_sim):
        start_time = time.time()

        print("Starting Race Game at time: {}".format(start_time))
        # Start the scripts independently

        stop_event = multiprocessing.Event()  # Event to signal when plot.py should stop

        processes = []
        processes.append(multiprocessing.Process(target=plot_independent, args=(GamePlanner, stop_event)))
        processes.append(multiprocessing.Process(target=GamePlanner.run_game, args=()))

        # Start all processes
        for process in processes:
            process.start()
        processes[1].join()  # Wait for the trajectory process to finish
        print("Trajectories found")
        
        time.sleep(2)
        stop_event.set()  # Event to signal when plot.py should stop

        for process in processes:
            process.join()

        # If main.py has finished, terminate all remaining processes
        for process in processes:
            if process.is_alive():
                process.terminate()

        duration = time.time() - start_time

        # Save duration to text file
        with open(os.path.join(path_to_results, GamePlanner.name + ".txt"), 'a') as f:
            f.write("Config: {}\n".format(GamePlanner.config))
            f.write(f"Duration: {duration}\n")
            f.write("\nContent of global_state.csv:\n")
            with open(os.path.join(path_to_data, "global_state.csv"), 'r') as csv_file:
                f.write(csv_file.read())

        print("Finished with duration: {} s".format(duration))

def run_experiment(exp_path_level_1, game_config, timestep_sim=None, exp_comment="", input=""):
    #print("Running MCTS in Experimental mode!")

    exp_path_level_0 = get_exp_path_level_0(exp_path_level_1, config=game_config, exp_comment=exp_comment, input=input)
    
    # WRITE CONFIGURATION FILE
    with open(os.path.join(exp_path_level_0, "config.json"), 'w') as f:
        json.dump(game_config, f)

    for run_ix in range(game_config.num_sim):
        GamePlanner = CompetitiveGame(Environment(game_config), game_config)
        exp_ix = run_ix % 10000
        subex_filepath= os.path.join(exp_path_level_0, str(exp_ix))

        if not os.path.exists(subex_filepath):
            os.mkdir(subex_filepath)

        # Redirect print outputs to a text file
        print_file = open(os.path.join(subex_filepath, "print_output.txt"), "w")
        sys.stdout = print_file

        # RUN EXPERIMENT
        result_dict, policy_dict = GamePlanner.run_game(timesteps_sim=timestep_sim)

        # PLOT TRAJECTORIES
        plot_trajectory(GamePlanner, result_dict, subex_filepath, all_timesteps=False)

        # Write the result dictionary to the JSON file
        with open(os.path.join(subex_filepath, "results.json"), "w") as f:
            json.dump(result_dict, f)
        with open(os.path.join(subex_filepath, "policies.json"), "w") as f:
            json.dump(policy_dict, f)
        
        # Restore the standard output
        sys.stdout = sys.__stdout__
        print_file.close()
    
    # COLLECT AND SAVE GLOBAL STATISTICAL DATA
    save_global_data(exp_path_level_0)
    save_statistical_data(exp_path_level_0, global_data="global_results.json")


def plot_trajectory(GamePlanner, result_dict, exp_path, all_timesteps=False):
    if all_timesteps:
        for t in range(result_dict['T_terminal']+1):
            plot_single_run(GamePlanner, exp_path, result_dict=result_dict, timestep=t, main_agent=0)
            plot_single_run(GamePlanner, exp_path, result_dict=result_dict, timestep=t, main_agent=1)
    else:
        plot_single_run(GamePlanner, exp_path, result_dict=result_dict, main_agent=0)
        plot_single_run(GamePlanner, exp_path, result_dict=result_dict, main_agent=1)

def create_global_index(exp_path_level_1):
    index_file_path = os.path.join(exp_path_level_1, "index.txt")
    if not os.path.exists(index_file_path):
        with open(index_file_path, 'w') as f:
            f.write("1")


def run_exp_vary_parameter(exp_path_level_1, game_dict, exp_params, timestep_sim=None):
    linspace = {}
    for parameter, param_range in exp_params.items():
        # build up linspace from (start, end, num) tuple
        linspace[parameter] = np.linspace(param_range[0], param_range[0] + param_range[1] * (param_range[2]-1), param_range[2])

    for param_values in itertools.product(*linspace.values()):
        update_dict = {parameter: value for parameter, value in zip(linspace.keys(), param_values)}
        exp_name = "_".join([f"{parameter}_{value}" for parameter, value in zip(linspace.keys(), param_values)])
        exp_config = Config(copy_new_dict(game_dict, update_dict))
        run_experiment(exp_path_level_1, game_config=exp_config, timestep_sim=timestep_sim, input=exp_name)


if __name__ == "__main__":
    
    experiments, global_exp_dir = exp_config.build_experiments()
    path_to_experiment = os.path.join(path_to_data,global_exp_dir)

    # Print all names of the list elements of experiments together with their indices
    for i, experiment in enumerate(experiments):
        print(f"{i}: {experiment['name']}")

    # Let the user choose which experiment to compute
    experiment_index = int(input("Enter the index of the experiment to compute: "))
    experiment = experiments[experiment_index]

    # RUN EXPERIMENT
    if not os.path.exists(path_to_experiment):
        os.mkdir(path_to_experiment)
    if experiment['dict']['feature_flags']["run_mode"]["exp"]:
        exp_path_level_1 = os.path.join(path_to_experiment, experiment['name'])

        if not os.path.exists(os.path.join(path_to_experiment, exp_path_level_1)):
            os.mkdir(os.path.join(path_to_experiment, exp_path_level_1))
        create_global_index(os.path.join(path_to_experiment, exp_path_level_1))
        
        exp_start = time.time()

        #SPECIFY MORE EXPERIMENT PARAMETERS TO INVESTIGATE
        exp_params = experiment['exp_params']
        run_exp_vary_parameter(exp_path_level_1, game_dict=experiment['dict'], exp_params=exp_params, timestep_sim=experiment['timestep_sim'])
        
        exp_duration = time.time() - exp_start

        print("Finished all experiments with duration: {} s".format(exp_duration))

    elif experiment['dict']['feature_flags']["run_mode"]["test"]:
        run_test(game_dict=experiment['dict'])
    
    print("All experiments finished")
   
