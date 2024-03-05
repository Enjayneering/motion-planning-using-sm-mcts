import yaml
import os
import json

path_to_exp_results = "/home/enjay/0_thesis/01_MCTS/experiments"


def find_json_file(directory, filename):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == filename:
                return os.path.join(root, file)
    return None

def read_json_entry(directory, filename, entry):
    filepath = find_json_file(directory, filename)
    if filepath:
        with open(filepath, "r") as file:
            data = json.load(file)
            return data.get(entry)
    else:
        print("JSON file not found.")
    """with open(file_path, "r") as file:
        data = json.load(file)
        return data.get(entry)"""

def read_json_file(directory, filename):
    file_path = find_json_file(directory, filename)
    if file_path:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data
    else:
        print("JSON file not found.")
    with open(file_path, "r") as file:
        data = json.load(file)
        return data

def round_dict(d, num=2):
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = round(v, num)  # round to num decimal places
        elif isinstance(v, dict):
            round_dict(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], float):
                    v[i] = round(v[i], num)  # round to num decimal places
                elif isinstance(v[i], dict):
                    round_dict(v[i])
    return d
