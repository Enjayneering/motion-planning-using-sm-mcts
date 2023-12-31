import subprocess
import time
import multiprocessing
import plot
import os

from common import *

def run_script(script, done_event=None, stop_event=None):
    if script == path_to_src+"plot.py":
        # Run plot.py in a separate process and pass the stop_event
        process = multiprocessing.Process(target=plot.main, args=(stop_event,))
        process.start()
    else:
        subprocess.run(["python3", script])
        if done_event:
            done_event.set()  # Signal that this script has finished

if __name__ == "__main__":
    start_time = time.time()

    # Create textfile for data
    with open(os.path.join(path_to_results, next_video_name + ".txt"), 'w') as f:
        for key, value in MCTS_params.items():
            f.write(f"{key}: {value}\n")
        for key, value in Competitive_params.items():
            f.write(f"{key}: {value}\n")

    print("Starting Race Game at time: {}".format(start_time))

    # Start the scripts independently
    processes = []

    done_event = multiprocessing.Event()  # Event to signal when main.py has finished
    stop_event = multiprocessing.Event()  # Event to signal when plot.py should stop
    processes.append(multiprocessing.Process(target=run_script, args=(path_to_src+"plot.py", done_event, stop_event)))
    processes.append(multiprocessing.Process(target=run_script, args=(path_to_src+"main.py", done_event, stop_event)))

    # Start all processes
    for process in processes:
        process.start()

    done_event.wait()  # Wait for main.py to finish
    print("Main.py has finished")

    stop_event.set()  # Event to signal when plot.py should stop

    for process in processes:
        process.join()

    # If main.py has finished, terminate all remaining processes
    for process in processes:
        if process.is_alive():
            process.terminate()

    duration = time.time() - start_time

    # Save duration to text file
    with open(os.path.join(path_to_results, next_video_name + ".txt"), 'a') as f:
        f.write(f"Duration: {duration}\n")

    print("Finished with duration: {} s".format(duration))