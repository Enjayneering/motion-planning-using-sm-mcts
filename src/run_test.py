import time
import multiprocessing
import os
import json
import itertools

# import modules
import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS')
from simulation import Simulation
from plot_independent import plot_independent
from utilities.plot_utilities import *
from utilities.common_utilities import *
from utilities.csv_utilities import *
from utilities.networkx_utilities import *
from utilities.run_utilities import *
from environment import *

from configs.config_global import *



def run_test(game_dict):
    Game = CompetitiveRace(Config(game_dict))
    for _ in range(Game.config.num_sim):
        start_time = time.time()

        print("Starting Race Game at time: {}".format(start_time))
        # Start the scripts independently

        stop_event = multiprocessing.Event()  # Event to signal when plot.py should stop

        processes = []
        processes.append(multiprocessing.Process(target=plot_independent, args=(Game, stop_event)))
        processes.append(multiprocessing.Process(target=run_simulation, args=()))

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

def plot_trajectory(Game, result_dict, exp_path, all_timesteps=False):
    if all_timesteps:
        for t in range(result_dict['T_terminal']+1):
            plot_single_run(Game, exp_path, result_dict=result_dict, timestep=t, main_agent=0)
            plot_single_run(Game, exp_path, result_dict=result_dict, timestep=t, main_agent=1)
    else:
        plot_single_run(Game, exp_path, result_dict=result_dict, main_agent=0)
        plot_single_run(Game, exp_path, result_dict=result_dict, main_agent=1)

def create_global_index(exp_path_level_1):
    index_file_path = os.path.join(exp_path_level_1, "index.txt")
    if not os.path.exists(index_file_path):
        with open(index_file_path, 'w') as f:
            f.write("1")



if __name__ == "__main__":
    # Setting test up

    run_test(config_global)
    
    print("All experiments finished")
   
