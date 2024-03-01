import csv
import shutil
import glob

# import modules
import os 
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

from utilities.common_utilities import *

#TODO: DONE

def get_state_fieldnames(config_global):
    fieldnames = []
    state_space = config_global['state_space']
    for agent_index in config_global['config_agents']:
        for state_element in state_space:
            fieldnames.append(str(state_element)+'_'+str(agent_index))
    return fieldnames

def csv_init_global_state(test_id, config_global):
    filepath = path_to_global_state.format(test_id)
    # removing all files from data
    """file_list = os.listdir(path_to_data)
    for file in file_list:
        file_path = os.path.join(path_to_data, file)
        os.remove(file_path)"""
    
    # writing global state data to csv file
    with open(filepath, mode='w') as csv_file:
        fieldnames = get_state_fieldnames(config_global)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def csv_write_global_state(test_id, config_global, state):
    return
    # update writing joint state
    filepath = path_to_global_state.format(test_id)

    with open(filepath, mode='a') as csv_file:
        fieldnames = get_state_fieldnames(config_global)
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        state_dict = {field: getattr(state, field) for field in fieldnames}
        writer.writerow(state_dict)

def csv_init_rollout_last(test_id, agent_ix, config_global):
    filepath_last = path_to_rollout_last.format(test_id, agent_ix)
    filepath_tmp = path_to_rollout_tmp.format(test_id, agent_ix)

    with open(filepath_last, mode='w') as csv_file:
        fieldnames = get_state_fieldnames(config_global)+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    with open(filepath_tmp, mode='w') as csv_file:
        fieldnames = get_state_fieldnames(config_global)+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

def csv_write_rollout_last(test_id, agent_ix, config_global, rollout_trajectory, timehorizon = None, keep_num_data=10000):
    return
    # update writing joint state
    # plot statistical exploration data
    filepath_temp = path_to_rollout_tmp.format(test_id, agent_ix)
    filepath_last = path_to_rollout_last.format(test_id, agent_ix)
    
    with open(filepath_temp, mode='a+') as csv_file:
        fieldnames = get_state_fieldnames(config_global)+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        for joint_state in rollout_trajectory:

            state_dict = {field: getattr(state, field) for field in get_state_fieldnames(config_global)}
            state_dict['timehorizon'] = timehorizon
            writer.writerow(state_dict)

        csv_file.seek(0)  # Move the file pointer to the beginning of the file
        lines = csv_file.readlines()[1:]

    lines = lines[-keep_num_data:]

    with open(filepath_temp, 'w') as csv_file:
        fieldnames = get_state_fieldnames(config_global)+['timehorizon']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.writelines(lines)
    shutil.copy(filepath_temp, filepath_last)

# Video parameters
def get_next_game_name(test_id, config_global):
    instance_name = config_global['config_environment']['name']
    filepath = os.path.join(path_to_results.format(test_id), instance_name + "*.mp4")
    list_of_files = glob.glob(filepath)
    num_videos = len(list_of_files)
    next_video_name = "{}_{:02d}".format(instance_name, num_videos + 1)
    return next_video_name

def write_config_txt(test_id, config_global):
    filepath = os.path.join(path_to_results.format(test_id), config_global['config_environment']['name'] + ".txt")
    with open(filepath, 'w') as f:
        for key, value in config_global.items():
            f.write(f"{key}: {value}\n")