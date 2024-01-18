import os
import json
import matplotlib.pyplot as plt
from common import path_to_experiments
from plot_utilities import coll_count

def plot_data(path_to_experiments, xaxis0, xaxis1, yaxis0, yaxis1):
    # Get the list of JSON files in the subfolder

    # Initialize lists to store x and y values for agent 0 and 1
    agent0_x = []
    agent0_y = []
    agent1_x = []
    agent1_y = []

    # choose the folders to plot from
    for folder in os.listdir(path_to_experiments):
        experiment_folder = os.path.join(path_to_experiments, folder)
    
        config_files = []
        results_files = []
        for root, dirs, files in os.walk(experiment_folder):
            for file in files:
                if file.endswith("config.json"):
                    config_files.append(os.path.join(root, file))
                if file.endswith("results.json"):
                    results_files.append(os.path.join(root, file))

        # Read the JSON files and extract the data
        for json_file in config_files:
            try:
                with open(os.path.join(experiment_folder, json_file)) as f:
                    data = json.load(f)
                    try:
                        agent0_x.append(data[xaxis0][-1])
                        agent1_x.append(data[xaxis1][-1])
                    except:
                        agent0_x.append(data[xaxis0])
                        agent1_x.append(data[xaxis0])
            except:
                pass
        for json_file in results_files:
            try:
                with open(os.path.join(experiment_folder, json_file)) as f:
                    data = json.load(f)
                    try:
                        agent0_y.append(data[yaxis0][-1])
                        agent1_y.append(data[yaxis1][-1])
                    except:
                        agent0_y.append(data[yaxis0])
                        agent1_y.append(data[yaxis1])
            except:
                pass
    
    # Check the minimum length of the lists
    min_length = min(len(agent0_x), len(agent0_y), len(agent1_x), len(agent1_y))
    print("agent0x: ", len(agent0_x))
    print("agent0y: ", len(agent0_y))
    print("agent1x: ", len(agent1_x))
    print("agent1y: ", len(agent1_y))

    # Delete the last elements of longer lists
    agent0_x = agent0_x[:min_length]
    agent0_y = agent0_y[:min_length]
    agent1_x = agent1_x[:min_length]
    agent1_y = agent1_y[:min_length]
    
    #print("agent0_x: ", agent0_x)

    # Plot the data
    plt.scatter(agent0_x, agent0_y, label="Agent 0")
    plt.scatter(agent1_x, agent1_y, label="Agent 1")
    
    plt.xlabel(xaxis0)
    plt.ylabel(yaxis1)
    plt.legend()
    plt.show()

def plot_weight_params(exp_subfolder):
    pen_dist_0 = []
    pen_dist_1 = []
    pen_time_0 = []
    pen_time_1 = []
    payoff_0 = []
    payoff_1 = []

    for folder in os.listdir(exp_subfolder):
        experiment_folder = os.path.join(exp_subfolder, folder)
    
        config_files = []
        results_files = []
        for root, dirs, files in os.walk(experiment_folder):
            for file in files:
                if file.endswith("config.json"):
                    config_files.append(os.path.join(root, file))
                if file.endswith("results.json"):
                    results_files.append(os.path.join(root, file))

        # Read the JSON files and extract the data
        for json_file in results_files:
            with open(os.path.join(experiment_folder, json_file)) as f:
                data = json.load(f)
                collisions = coll_count(data["trajectories"])
                print(collisions)
                if collisions == 0:
                    payoff_0.append(data["payoff_total0"][-1])
                    payoff_1.append(data["payoff_total1"][-1])
                    for json_file in config_files:
                        try:
                            with open(os.path.join(experiment_folder, json_file)) as f:
                                data = json.load(f)
                                try:
                                    pen_dist_0.append(data["penalty_distance_0"])
                                    pen_dist_1.append(data["penalty_distance_1"])
                                    pen_time_0.append(data["penalty_timestep_0"])
                                    pen_time_1.append(data["penalty_timestep_1"])
                                except:
                                    pass
                        except:
                            pass

    plt.scatter(pen_dist_0, 
                pen_time_0, 
                marker="o", 
                c=[x+y for x, y in zip(payoff_0, payoff_1)], 
                cmap=plt.cm.coolwarm)

    # Label each datapoint with the number of collisions
    #for i, num_collisions in enumerate(num_coll):
    #    plt.text(pen_dist_0[i], pen_time_0[i], str(num_collisions), ha='center', va='bottom')

    plt.xlabel("penalty_distance")
    plt.ylabel("penalty_timestep")
    plt.colorbar(label="C Value")
    plt.show()


"""def process_experiments(path_to_experiments):
    # Get the list of folders in the experiments directory
    folders = [folder for folder in os.listdir(path_to_experiments) if os.path.isdir(os.path.join(path_to_experiments, folder))]
    
    # Initialize a counter for the unique integer index
    index = 0
    
    # Iterate through each folder
    for folder in folders:
        folder_path = os.path.join(path_to_experiments, folder)

        
        # Check if the folder contains result files
        if has_result_files(folder_path):
            # Rename the folder with the unique integer index
            new_folder_name = str(index)+str(folder)
            new_folder_path = os.path.join(path_to_experiments, new_folder_name)
            print("Renaming folder: ", folder_path, " to ", new_folder_path)
            #os.rename(folder_path, new_folder_path)
            
            # Increment the index counter
            index += 1
        else:
            # Delete the folder if it doesn't have result files
            print("Deleting folder: ", folder_path)
            #os.rmdir(folder_path)

def has_result_files(folder_path):
    # Check if the folder contains result files
    result_files = [file for file in os.listdir(folder_path) if file.endswith("results.json")]
    return len(result_files) > 0"""





xaxis0 = "penalty_timestep_0" #"penalty_distance_0"
xaxis1 = "penalty_timestep_0" #"penalty_distance_1"
yaxis0 = "payoff_total0"
yaxis1 = "payoff_total1"

#plot_data(path_to_experiments, xaxis0, xaxis1, yaxis0, yaxis1)

plot_weight_params(os.path.join(path_to_experiments, "01_weights_param_study"))