import os
import statistics

import pandas as pd

# change to parent dict for import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from process_data_utilities import *

path_to_data = "/home/enjay/0_thesis/01_MCTS/data/"
name_exp = "01_convergence_overtaking_single_mcts"

processed_data = {} # {action: [iter, payoff]}

if __name__ == "__main__":
    path_curr_experiment = os.path.join(path_to_data, name_exp)
    path_raw = os.path.join(path_curr_experiment, "raw")
    path_processed = os.path.join(path_curr_experiment, "processed")
    processed_data = {}

    # Iterate over all subfolders in the specified folder
    
    for game_length in range(1,15):
        exp_payoff_mean_calc = {}
        for subfolder_iter in os.listdir(path_raw):
            subfolder_iter_path = os.path.join(path_raw, subfolder_iter)
            # Initialize dictionary to store expected payoff mean for each action
        


            if read_json_entry(directory=subfolder_iter_path, filename="results.json", entry="max_timestep") == game_length:
                if os.path.isdir(subfolder_iter_path):

                    num_iter = read_json_entry(directory=subfolder_iter_path, filename="config.json", entry="num_iter")
                    mean_runtime = round(read_json_entry(directory=subfolder_iter_path, filename="global_results_statistical.json", entry="runtime_gamelength_{}".format(game_length))['mean'], 2)

                    config = read_json_file(directory=subfolder_iter_path, filename="config.json")
                    #print("config: ", config)
                    #print("mean_runtime: ", mean_runtime)
                    #print("num_iter: ", num_iter)

                    for subfolder_run in os.listdir(subfolder_iter_path):
                        subfolder_run_path = os.path.join(subfolder_iter_path, subfolder_run)

                        if os.path.isdir(subfolder_run_path):
                            # add alpha_t to the list
                            results = read_json_file(directory=subfolder_run_path, filename="results.json")
                            #var_payoff_total = read_json_entry(directory=subfolder_iter_path, filename="global_results_statistical.json", entry="payoff_total")['variance'][-1][-1]
                            #alpha_t_mean_calc.append(compute_alpha_t(results, config, game_length))

                            # Read the JSON files and extract the data
                            input_dict_gamelength = read_json_entry(directory=subfolder_run_path, filename="policies.json", entry=str(game_length))
                            #print("input_dict: ", input_dict)

                            # go through agents' policies
                            for agent, policy_dict in input_dict_gamelength.items():
                                agent = int(agent)
                                if exp_payoff_mean_calc.get(agent) is None:
                                    exp_payoff_mean_calc[agent] = {}
                                if processed_data.get(agent) is None:
                                    processed_data[agent] = {}
                                for action, policy_data in policy_dict.items():
                                    if exp_payoff_mean_calc[agent].get(action) is None:
                                        exp_payoff_mean_calc[agent][action] = {}
                                    if exp_payoff_mean_calc[agent][action].get((num_iter, mean_runtime)) is None:
                                        exp_payoff_mean_calc[agent][action][(num_iter, mean_runtime)] = []
                                    exp_payoff_mean_calc[agent][action][(num_iter, mean_runtime)].append(policy_data['sum_payoffs']/policy_data['num_count'])

        # calculate mean expected payoff for each action
        for agent, exp_payoff_mean_calc_action in exp_payoff_mean_calc.items():
            #print("exp_payoff_mean_calc_action: ", exp_payoff_mean_calc_action)
            for action, list_iter in exp_payoff_mean_calc_action.items():
                for (num_iter, mean_runtime), exp_payoffs in list_iter.items():
                    exp_payoff_mean = sum(exp_payoffs) / len(exp_payoffs)
                    exp_payoff_var = statistics.variance(exp_payoffs)
                    if processed_data[agent].get(action) is None:
                        processed_data[agent][action] = {}
                    #if processed_data[agent][action].get(num_iter) is None:
                    #    processed_data[agent][action][num_iter] = []
                    processed_data[agent][action][num_iter] = [mean_runtime, exp_payoff_mean, exp_payoff_var]

        # calculate mean alpha_t
        #print('alpha_t: ', alpha_t_mean_calc)
        #alpha_t_mean = sum(alpha_t_mean_calc) / len(alpha_t_mean_calc)

        # Sort data to increasing iter values
        sorted_processed_data = {}
        for agent, agent_data in processed_data.items():
            sorted_processed_data[agent] = {}
            for action, list_iter in agent_data.items():
                sorted_processed_data[agent][action] = {}
                sorted_list_iter = sorted(list_iter.items(), key=lambda x: x[0])
                for num_iter, action_data in sorted_list_iter:
                    sorted_processed_data[agent][action][num_iter] = action_data

        processed_data = sorted_processed_data
        #print("processed_data: ", processed_data)

        # store in pandas data frame
        # Flatten the dictionary
        data = []
        for agent, actions in processed_data.items():
            for action, num_iters in actions.items():
                for num_iter, values in num_iters.items():
                    row = {'agent': agent, 'action': action, 'num_iter': num_iter}
                    row.update(dict(zip(['runtime', 'payoff_mean', 'payoff_var'], values)))
                    data.append(row)

        # Create a DataFrame
        df = pd.DataFrame(data)
        print(df)

        # save pandas dataframe to csv
        df.to_csv(os.path.join(path_processed, "gamelength_{}.csv".format(game_length)), index=False)
