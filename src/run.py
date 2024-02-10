import time
import multiprocessing
import os
import json
import csv

from competitive_game import CompetitiveGame
from plot_independent import plot_independent
from plot_utilities import *
from common import *
from csv_utilities import *
from config import *
from run_utilities import *
from environment_utilities import *


def run_test(game_config):
    Game = CompetitiveGame(game_config)
    for _ in range(Game.config.num_sim):
        start_time = time.time()

        print("Starting Race Game at time: {}".format(start_time))
        # Start the scripts independently

        stop_event = multiprocessing.Event()  # Event to signal when plot.py should stop

        processes = []
        processes.append(multiprocessing.Process(target=plot_independent, args=(Game, stop_event)))
        processes.append(multiprocessing.Process(target=Game.run_simulation, args=()))

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
        with open(os.path.join(path_to_results, Game.name + ".txt"), 'a') as f:
            f.write("Config: {}\n".format(Game.config))
            f.write(f"Duration: {duration}\n")
            f.write("\nContent of global_state.csv:\n")
            with open(os.path.join(path_to_data, "global_state.csv"), 'r') as csv_file:
                f.write(csv_file.read())

        print("Finished with duration: {} s".format(duration))

def run_experiment(exp_path_level_1, game_config, exp_comment="", input=""):
    print("Running MCTS in Experimental mode!")

    exp_path_level_0 = get_exp_path_level_0(exp_path_level_1, config=game_config, exp_comment=exp_comment, input=input)
    
    # WRITE CONFIGURATION FILE
    with open(os.path.join(exp_path_level_0, "config.json"), 'w') as f:
        json.dump(game_config, f)

    for run_ix in range(game_config.num_sim):
        Game = CompetitiveGame(game_config)
        exp_ix = run_ix % 10000
        subex_filepath= os.path.join(exp_path_level_0, str(exp_ix))

        if not os.path.exists(subex_filepath):
            os.mkdir(subex_filepath)

        # Redirect print outputs to a text file
        print_file = open(os.path.join(subex_filepath, "print_output.txt"), "w")
        sys.stdout = print_file

        # RUN EXPERIMENT
        result_dict = Game.run_simulation()

        # PLOT TRAJECTORIES
        plot_trajectory(Game, result_dict, subex_filepath, all_timesteps=False)

        # Write the result dictionary to the JSON file
        with open(os.path.join(subex_filepath, "results.json"), "w") as f:
            json.dump(result_dict, f)
        
        # Restore the standard output
        sys.stdout = sys.__stdout__
        print_file.close()
    
    # COLLECT AND SAVE GLOBAL STATISTICAL DATA
    save_global_data(exp_path_level_0)
    save_statistical_data(exp_path_level_0, global_data="global_results.json")


def plot_trajectory(Game, result_dict, exp_path, all_timesteps=False):
    if all_timesteps:
        for t in range(result_dict['T_terminal']+1):
            plot_single_run(Game, result_dict, exp_path, timestep=t, main_agent=0)
            plot_single_run(Game, result_dict, exp_path, timestep=t, main_agent=1)
    else:
        plot_single_run(Game, result_dict, exp_path, main_agent=0)
        plot_single_run(Game, result_dict, exp_path, main_agent=1)

def create_global_index(exp_path_level_1):
    with open(os.path.join(exp_path_level_1, "index.txt"), 'w') as f:
        f.write("1")

def run_exp_vary_parameter(exp_path_level_1, game_config, parameter, linspace, dtype="float"):
    num_incr = linspace[2]
    if dtype == "float":
        for param_value in np.linspace(linspace[0], linspace[1], num_incr):
            update_dict = {parameter: param_value}
            exp_config = copy_new_config(game_config, update_dict, env_dict, env_name)
            run_experiment(exp_path_level_1, game_config=exp_config, input=f"{parameter}_{param_value}")
    elif dtype == "int":
        for param_value in np.linspace(linspace[0], linspace[1], num_incr):
            update_dict = {parameter: int(param_value)}
            exp_config = copy_new_config(game_config, update_dict, env_dict, env_name)
            run_experiment(exp_path_level_1, game_config=exp_config, input=f"{param_value}")


if __name__ == "__main__":
    # SETTING EXPERIMENT UP
    expdict = curr_dict

    exp_path_level_2 = 'V5_Test_coll_pruning'

    config = copy_new_config(default_config, expdict, env_dict)


    if config.feature_flags["run_mode"]["exp"]:
        exp_path_level_1 = os.path.join(path_to_experiments, exp_path_level_2)

        if not os.path.exists(os.path.join(path_to_experiments, exp_path_level_1)):
            os.mkdir(os.path.join(path_to_experiments, exp_path_level_1))
        create_global_index(os.path.join(path_to_experiments, exp_path_level_1))
        
        exp_start = time.time()

        run_experiment(exp_path_level_1, game_config=config, input=str(config.num_iter))
        
        #run_exp_vary_parameter(exp_path_level_1, game_config=config, parameter='c_param', linspace=(100, 1000, 10), dtype="float")

        """num_incr = 10
        for incr_dist in range(0, num_incr+1):
            for incr_timestep in range(0, num_incr+1):
                increment_dict = {  'penalty_distance_0': -incr_dist/num_incr,
                                    'penalty_distance_1': -incr_dist/num_incr,
                                    'penalty_timestep_0': -incr_timestep/num_incr,
                                    'penalty_timestep_1': -incr_timestep/num_incr}
                exp_weights_increment_config = copy_new_config(game_config, increment_dict)
                #print(exp_weights_increment_config)

                # RUN EXPERIMENT
                run_experiment(exp_path_level_1, game_config=config, input=f"{incr_dist/num_incr}-{incr_timestep/num_incr}")"""

        
        exp_duration = time.time() - exp_start
        print("Finished all experiments with duration: {} s".format(exp_duration))

    elif config.feature_flags["run_mode"]["test"]:
        run_test(game_config=config)
    
    
   
