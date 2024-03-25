import os
import json

from statistics import mean, variance

def get_exp_path_level_0(exp_subfolder, config, exp_comment="", input=""):
    #with open(os.path.join(exp_subfolder, "index.txt"), 'r') as f:
    #    global_index = int(f.read())
    
    # INITIALIZE FOLDER
    #start_config = "dis" # sym: "symmetric", adv: "advantageous", dis: "disadvantageous" (perspective of Agent 0)
    selection = [flag for flag in config.feature_flags["selection_policy"] if config.feature_flags["selection_policy"][flag] == True][0]
    final_move_selection = [flag for flag in config.feature_flags["final_move"] if config.feature_flags["final_move"][flag] == True][0]
    rollout_policy = [flag for flag in config.feature_flags["rollout_policy"] if config.feature_flags["rollout_policy"][flag] == True][0]
    exp_name = f"{exp_comment}_{input}_{selection}_{final_move_selection}_{rollout_policy}"

    exp_filepath = os.path.join(exp_subfolder, exp_name)

    if not os.path.exists(exp_filepath):
        os.mkdir(exp_filepath)

    #with open(os.path.join(exp_subfolder, "index.txt"), 'w') as f:
    #    global_index += 1
    #    f.write(str(global_index))
    return exp_filepath

def save_global_data(exp_path_level_0):
    # read all data from result files in subfolders
    result_data = _read_result_files(exp_path_level_0) # list
    
    # Save the result data to a JSON file
    with open(os.path.join(exp_path_level_0, "global_results.json"), 'w') as json_file:
        json.dump(result_data, json_file)

def save_statistical_data(exp_path_level_0, global_data="global_results.json"):
    # open global data file
    with open(os.path.join(exp_path_level_0, global_data), 'r') as json_file:
        data_columns = json.load(json_file)
    
    data_columns_statistical = {}

    for key in data_columns.keys():
        # extract mean and variance
        try: # if list elements are single values
            mean_value = _mean_value(data_columns[key])
            variance_value = _variance_value(data_columns[key])
            data_columns_statistical[key] = {"mean": mean_value, "variance": variance_value}
        except: # if list elements are lists
            try: # if list elements are single values
                mean_list = _mean_list_of_lists(data_columns[key])
                variance_list = _variance_list_of_lists(data_columns[key])
                data_columns_statistical[key] = {"mean": mean_list, "variance": variance_list}
            except: # if list elements are lists of lists
                mean_list_of_lists = _mean_list_of_lists_of_lists(data_columns[key])
                variance_list_of_lists = _variance_list_of_lists_of_lists(data_columns[key])
                data_columns_statistical[key] = {"mean": mean_list_of_lists, "variance": variance_list_of_lists}

    # write to JSON file
    with open(os.path.join(exp_path_level_0, "global_results_statistical.json"), 'w') as json_file:
        json.dump(data_columns_statistical, json_file)


def _read_result_files(exp_path_level_0):
    result_data = {}
    for folder_name in os.listdir(exp_path_level_0):
        folder_path_single_run = os.path.join(exp_path_level_0, folder_name)
        
        if os.path.isdir(folder_path_single_run):
            for file_name in os.listdir(folder_path_single_run):
                if file_name.endswith("results.json"):
                    file_path = os.path.join(folder_path_single_run, file_name)
                    
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        for key, value in data.items():
                            if key not in result_data:
                                result_data[key] = []
                            result_data[key].append(value)
    return result_data

def _mean_value(data_as_list):
    return sum([d for d in data_as_list])/len(data_as_list)

def _variance_value(data_as_list):
    mean = _mean_value(data_as_list)
    return sum([(d-mean)**2 for d in data_as_list])/len(data_as_list)

def _mean_list_of_lists(data_list):
    sub_mean = [sum(sub_list)/len(sub_list) for sub_list in zip(*data_list)]
    return sub_mean

def _mean_list_of_lists_of_lists(data_list):
    sub_sub_mean = [[sum(sub_sub_list)/len(sub_sub_list) for sub_sub_list in zip(*sub_list)] for sub_list in zip(*data_list)]
    return sub_sub_mean

def _variance_list_of_lists(data_list):
    sub_mean = _mean_list_of_lists(data_list)
    sub_variance = [sum([(d-sub_mean[i])**2 for d in sub_list])/len(sub_list) for i, sub_list in enumerate(zip(*data_list))]
    return sub_variance

def _variance_list_of_lists_of_lists(data_list):
    sub_sub_mean = _mean_list_of_lists_of_lists(data_list)
    sub_sub_variance = [[sum([(d-sub_sub_mean[i][j])**2 for d in sub_sub_list])/len(sub_sub_list) for j, sub_sub_list in enumerate(zip(*sub_list))] for i, sub_list in enumerate(zip(*data_list))]
    return sub_sub_variance



