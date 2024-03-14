import csv
import os
import shutil
import glob

from .common_utilities import *

def csv_init_global_state(Game):
    # writing global state data to csv file
    with open(path_to_global_state, mode='w') as csv_file:
        fieldnames = Game.Model_params["state_space"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def csv_write_global_state(Game, state):
    with open(path_to_global_state, mode='a') as csv_file:
        fieldnames = Game.Model_params["state_space"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        state_dict = {field: getattr(state, field) for field in fieldnames}
        writer.writerow(state_dict)

def csv_init_rollout_last(Game):
    with open(path_to_rollout_last, mode='w') as csv_file:
        fieldnames = Game.Model_params["state_space"]+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    with open(path_to_rollout_tmp, mode='w') as csv_file:
        fieldnames = Game.Model_params["state_space"]+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def csv_write_rollout_last(Game, rollout_trajectory, timehorizon = None, config=None):
    # plot statistical exploration data
    with open(path_to_rollout_tmp, mode='a+') as csv_file:
        fieldnames = Game.Model_params["state_space"]+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for state in rollout_trajectory:
            state_dict = {field: getattr(state, field) for field in Game.Model_params["state_space"]}
            state_dict['timehorizon'] = timehorizon
            writer.writerow(state_dict)
        csv_file.seek(0)  # Move the file pointer to the beginning of the file
        lines = csv_file.readlines()[1:]

    keep_num_data = int(config.num_iter*get_max_timehorizon(Game)/freq_stat_data)+1
    lines = lines[-keep_num_data:]

    with open(path_to_rollout_tmp, 'w') as csv_file:
        fieldnames = Game.Model_params["state_space"]+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.writelines(lines)

    shutil.copy(path_to_rollout_tmp, path_to_rollout_last)

# Video parameters
def get_next_game_name(path_to_results, name):
    list_of_files = glob.glob(path_to_results + name + "*.mp4")
    num_videos = len(list_of_files)
    next_video_name = "{}_{:02d}".format(name, num_videos + 1)
    return next_video_name


def test_write_params(Game):
    with open(os.path.join(path_to_results, Game.name + ".txt"), 'w') as f:
        for key, value in Game.Model_params.items():
            f.write(f"{key}: {value}\n")
        for key, value in Game.MCTS_params.items():
            f.write(f"{key}: {value}\n")
        for key, value in Game.Kinodynamic_params.items():
            f.write(f"{key}: {value}\n") 