import time
import multiprocessing
import os

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
        processes.append(multiprocessing.Process(target=Game.find_nash_strategies, args=()))

        # Start all processes
        for process in processes:
            process.start()
        processes[1].join()  # Wait for the trajectory process to finish
        print("Trajectories found")
        
        time.sleep(1)
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
            f.write(f"Duration: {duration}\n")
            f.write("\nContent of global_state.csv:\n")
            with open(os.path.join(path_to_data, "global_state.csv"), 'r') as csv_file:
                f.write(csv_file.read())

        print("Finished with duration: {} s".format(duration))


def run_experiment(Game):
    # Init global csv configuration

    for run_ix in range(Game.config.num_sim):
        # Print Local csv 

        start_time = time.time()
        Game.find_nash_strategies()
        # save local data to csv (incl detailed payoff data)
        # plot found trajectories

    # save global statistical data to csv
    
if __name__ == "__main__": 
    if is_feature_active(feature_flags["mode"]["experimental"]):
        Game = CompetitiveGame(exp_config)
        run_experiment(Game)

    elif is_feature_active(feature_flags["mode"]["test"]):
        Game = CompetitiveGame(test_config)
        run_test(Game)
    
    
   
