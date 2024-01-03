import csv
import os
import shutil

from common import *
from environment import *

def csv_init_global_state():
    # removing all files from data
    """file_list = os.listdir(path_to_data)
    for file in file_list:
        file_path = os.path.join(path_to_data, file)
        os.remove(file_path)"""

    # writing global state data to csv file
    with open(path_to_global_state, mode='w') as csv_file:
        fieldnames = state_space
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def csv_write_global_state(state):
    with open(path_to_global_state, mode='a') as csv_file:
        fieldnames = state_space
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        state_dict = {field: getattr(state, field) for field in fieldnames}
        writer.writerow(state_dict)


"""def csv_init_rollout_curr():
    with open(path_to_rollout_curr, mode='w') as csv_file:
        fieldnames = state_space+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()"""

def csv_init_rollout_last():
    with open(path_to_rollout_last, mode='w') as csv_file:
        fieldnames = state_space+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    with open(path_to_rollout_tmp, mode='w') as csv_file:
        fieldnames = state_space+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def csv_write_rollout_last(rollout_trajectory, timehorizon = None):
    # plot statistical exploration data
    with open(path_to_rollout_tmp, mode='a+') as csv_file:
        fieldnames = state_space+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for state in rollout_trajectory:
            state_dict = {field: getattr(state, field) for field in state_space}
            state_dict['timehorizon'] = timehorizon
            writer.writerow(state_dict)
        csv_file.seek(0)  # Move the file pointer to the beginning of the file
        lines = csv_file.readlines()[1:]

    keep_num_data = int(MCTS_params['num_iter']*env.max_timehorizon/freq_stat_data)+1
    lines = lines[-keep_num_data:]

    with open(path_to_rollout_tmp, 'w') as csv_file:
        fieldnames = state_space+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.writelines(lines)

    shutil.copy(path_to_rollout_tmp, path_to_rollout_last)