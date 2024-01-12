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
        processes.append(multiprocessing.Process(target=Game.find_nash_strategies, args=()))

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
            f.write(f"Duration: {duration}\n")
            f.write("\nContent of global_state.csv:\n")
            with open(os.path.join(path_to_data, "global_state.csv"), 'r') as csv_file:
                f.write(csv_file.read())

        print("Finished with duration: {} s".format(duration))


def run_experiment(Game):
    print("Running MCTS in Experimental mode!")
    # INITIALIZE FOLDER
    param_of_investigation = "w" # w: "weights", t: "T_max/T_goal", n: "num_iter"
    start_config = "s" # s: "symmetric", a: "advantageous", d: "disadvantageous" (perspective of Agent 0)
    selection = [flag for flag in Game.config.feature_flags["selection_policy"] if Game.config.feature_flags["selection_policy"][flag] == True][0]
    final_move_selection = [flag for flag in Game.config.feature_flags["final_move"] if Game.config.feature_flags["final_move"][flag] == True][0]
    rollout_policy = [flag for flag in Game.config.feature_flags["rollout_policy"] if Game.config.feature_flags["rollout_policy"][flag] == True][0]
    environment = Game.config.env_name_trigger[0][1]
    exp_name = f"{param_of_investigation}_{start_config}_{selection}_{final_move_selection}_{rollout_policy}_{environment}"

    # Check if folder with the same exp_name already exists
    index = 0
    while os.path.exists(os.path.join(path_to_experiments, f"{exp_name}_{index:02d}")):
        index += 1
    exp_name = f"{exp_name}_{index:02d}"

    exp_filepath = os.path.join(path_to_experiments, exp_name)
    os.mkdir(exp_filepath)

    # WRITE CONFIGURATION FILE
    with open(os.path.join(exp_filepath, "config.json"), 'w') as f:
        json.dump(Game.config.__dict__, f)

    for run_ix in range(Game.config.num_sim):
        exp_ix = run_ix % 10000

        os.mkdir(os.path.join(exp_filepath, str(exp_ix)))

        # RUN EXPERIMENT
        result_dict = Game.find_nash_strategies()

        # Write the result dictionary to the JSON file
        with open(os.path.join(exp_filepath, str(exp_ix), "results.json"), "w") as f:
            json.dump(result_dict, f)

    # COLLECT AND SAVE GLOBAL STATISTICAL DATA



if __name__ == "__main__": 
    if experimental_mode:
        Game = CompetitiveGame(exp001_config)
        run_experiment(Game)

    else:
        Game = CompetitiveGame(test_config)
        run_test(Game)
    
    
   
