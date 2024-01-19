import time
import multiprocessing
import os
import json

from competitive_game import CompetitiveGame
from plot_independent import plot_independent
from plot_utilities import *
from common import *
from csv_utilities import *
from config import *
from environment_utilities import *

def run_test(Game):
    for _ in range(Game.config.num_sim):
        start_time = time.time()

        print("Starting Race Game at time: {}".format(start_time))
        # Start the scripts independently

        stop_event = multiprocessing.Event()  # Event to signal when plot.py should stop

        processes = []
        processes.append(multiprocessing.Process(target=plot_independent, args=(Game, stop_event)))
        processes.append(multiprocessing.Process(target=Game.compute_trajectories, args=()))

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


def run_experiment(exp_subfolder, Game, exp_comment="", input=""):
    print("Running MCTS in Experimental mode!")

    exp_filepath = get_exp_filepath(exp_subfolder, exp_comment, input)

    # WRITE CONFIGURATION FILE
    with open(os.path.join(exp_filepath, "config.json"), 'w') as f:
        json.dump(Game.config, f)

    for run_ix in range(Game.config.num_sim):
        
        exp_ix = run_ix % 10000
        subex_filepath= os.path.join(exp_filepath, str(exp_ix))

        if not os.path.exists(subex_filepath):
            os.mkdir(subex_filepath)

        # Redirect print outputs to a text file
        print_file = open(os.path.join(subex_filepath, "print_output.txt"), "w")
        sys.stdout = print_file

        # RUN EXPERIMENT
        result_dict = Game.compute_trajectories()

        # PLOT TRAJECTORIES
        plot_trajectory(result_dict, subex_filepath, all_timesteps=False)

        # Write the result dictionary to the JSON file
        with open(os.path.join(subex_filepath, "results.json"), "w") as f:
            json.dump(result_dict, f)

        # Restore the standard output
        sys.stdout = sys.__stdout__
        print_file.close()

    # COLLECT AND SAVE GLOBAL STATISTICAL DATA

def get_exp_filepath(exp_subfolder, exp_comment="", input=""):
    with open(os.path.join(exp_subfolder, "index.txt"), 'r') as f:
        global_index = int(f.read())
    
    # INITIALIZE FOLDER
    start_config = "dis" # sym: "symmetric", adv: "advantageous", dis: "disadvantageous" (perspective of Agent 0)
    num_iter = Game.config.num_iter
    selection = [flag for flag in Game.config.feature_flags["selection_policy"] if Game.config.feature_flags["selection_policy"][flag] == True][0]
    final_move_selection = [flag for flag in Game.config.feature_flags["final_move"] if Game.config.feature_flags["final_move"][flag] == True][0]
    rollout_policy = [flag for flag in Game.config.feature_flags["rollout_policy"] if Game.config.feature_flags["rollout_policy"][flag] == True][0]
    env_name = Game.config.env_name
    exp_name = f"{global_index}_{exp_comment}_{input}_{start_config}_{num_iter}_{selection}_{final_move_selection}_{rollout_policy}_{env_name}"

    exp_filepath = os.path.join(exp_subfolder, exp_name)

    if not os.path.exists(exp_filepath):
        os.mkdir(exp_filepath)

    with open(os.path.join(exp_subfolder, "index.txt"), 'w') as f:
        global_index += 1
        f.write(str(global_index))
    return exp_filepath

def plot_trajectory(result_dict, exp_path, all_timesteps=False):
    if all_timesteps:
        for t in range(result_dict['T_terminal']+1):
            plot_single_run(Game.config, result_dict, exp_path, timestep=t, main_agent=0)
            plot_single_run(Game.config, result_dict, exp_path, timestep=t, main_agent=1)
    else:
        plot_single_run(Game.config, result_dict, exp_path, main_agent=0)
        plot_single_run(Game.config, result_dict, exp_path, main_agent=1)

def create_global_index(exp_subfolder):
    with open(os.path.join(exp_subfolder, "index.txt"), 'w') as f:
        f.write("1")

if __name__ == "__main__": 
    if experimental_mode:
        exp_subfolder = "04_test"
        if not os.path.exists(os.path.join(path_to_experiments, exp_subfolder)):
            os.mkdir(os.path.join(path_to_experiments, exp_subfolder), )
        create_global_index(os.path.join(path_to_experiments, exp_subfolder))
        
        exp_start = time.time()
        num_incr = 10
        for incr_dist in range(0, num_incr+1):
            for incr_timestep in range(0, num_incr+1):
                increment_dict = {  'penalty_distance_0': -incr_dist/num_incr,
                                    'penalty_distance_1': -incr_dist/num_incr,
                                    'penalty_timestep_0': -incr_timestep/num_incr,
                                    'penalty_timestep_1': -incr_timestep/num_incr}
                exp_weights_increment_config = copy_new_config(exp_weights_config, increment_dict)
                print(exp_weights_increment_config)
                Game = CompetitiveGame(exp_weights_increment_config)
                run_experiment(os.path.join(path_to_experiments, exp_subfolder), Game, input=f"{incr_dist/num_incr}-{incr_timestep/num_incr}")
        exp_duration = time.time() - exp_start
        print("Finished all experiments with duration: {} s".format(exp_duration))

    else:
        Game = CompetitiveGame(test_config)
        run_test(Game)
    
    
   
