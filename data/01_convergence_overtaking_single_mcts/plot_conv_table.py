import sys
import os

# change to parent dict for import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from process_data_utilities import *
sys.path.append('/home/enjay/0_thesis/01_MCTS/src')
from plot_data_utilities import *
from utilities.env_utilities import *

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LinearRegression
import re
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import statistics
import os
import numpy as np
import statistics

#############
# Compute alpha_t if it wasn't saved in the results
def find_highest_number(dictionary, string):
    pattern = r"\d+"
    highest_number = 0

    for key in dictionary.keys():
        match = re.search(pattern, key)
        if match and string in key:
            number = int(match.group())
            highest_number = max(highest_number, number)

    return highest_number

def get_min_time_to_complete_modified(curr_state=None, goal_state=None, config=None, agent= None):
        min_times = []

        if agent == 0 or agent == None:
            dist_0 = distance(curr_state[0:2], goal_state[0:2])
            max_velocity_0 = np.max(config["velocity_0"])
            min_times.append(dist_0/max_velocity_0)
        if agent == 1 or agent == None:
            dist_1 = distance(curr_state[4:6], goal_state[0:2])
            max_velocity_1 = np.max(config["velocity_1"])
            min_times.append(dist_1/max_velocity_1)    
        return max(min_times)

def compute_alpha_t(results, config, game_length):
    max_timehorizon = find_highest_number(results, string='runtime_game_length_')
    ix_curr_state = max_timehorizon-game_length
    curr_state_0 = results['trajectory_0'][ix_curr_state]
    curr_state_1 = results['trajectory_1'][ix_curr_state]
    curr_state = curr_state_0 + curr_state_1
    min_time_to_goal = get_min_time_to_complete_modified(curr_state, goal_state, config)
    alpha_t = game_length/min_time_to_goal
    return alpha_t

def plot_figure(ax1, ax2, path_curr_experiment, game_length):
    # Extract Expected Payoff Data over different number of iterations
    data_to_plot = {} # {action: [iter, payoff]}

    # Iterate over all subfolders in the specified folder
    exp_payoff_mean_calc = {}
    for subfolder_iter in os.listdir(path_curr_experiment):
        subfolder_iter_path = os.path.join(path_curr_experiment, subfolder_iter)
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
                            if data_to_plot.get(agent) is None:
                                data_to_plot[agent] = {}
                            for action, policy_data in policy_dict.items():
                                if exp_payoff_mean_calc[agent].get(action) is None:
                                    exp_payoff_mean_calc[agent][action] = {}
                                if exp_payoff_mean_calc[agent][action].get((num_iter, mean_runtime)) is None:
                                    exp_payoff_mean_calc[agent][action][(num_iter, mean_runtime)] = []
                                exp_payoff_mean_calc[agent][action][(num_iter, mean_runtime)].append(policy_data['sum_payoffs']/policy_data['num_count'])
                result_dict_example = results



    # calculate mean expected payoff for each action
    for agent, exp_payoff_mean_calc_action in exp_payoff_mean_calc.items():
        #print("exp_payoff_mean_calc_action: ", exp_payoff_mean_calc_action)
        for action, list_iter in exp_payoff_mean_calc_action.items():
            for (num_iter, mean_runtime), exp_payoffs in list_iter.items():
                exp_payoff_mean = sum(exp_payoffs) / len(exp_payoffs)
                exp_payoff_var = statistics.variance(exp_payoffs)
                if data_to_plot[agent].get(action) is None:
                    data_to_plot[agent][action] = {}
                #if data_to_plot[agent][action].get(num_iter) is None:
                #    data_to_plot[agent][action][num_iter] = []
                data_to_plot[agent][action][num_iter] = [mean_runtime, exp_payoff_mean, exp_payoff_var]

    # calculate mean alpha_t
    #print('alpha_t: ', alpha_t_mean_calc)
    #alpha_t_mean = sum(alpha_t_mean_calc) / len(alpha_t_mean_calc)

    # Sort data to increasing iter values
    sorted_data_to_plot = {}
    for agent, agent_data in data_to_plot.items():
        sorted_data_to_plot[agent] = {}
        for action, list_iter in agent_data.items():
            sorted_data_to_plot[agent][action] = {}
            sorted_list_iter = sorted(list_iter.items(), key=lambda x: x[0])
            for num_iter, action_data in sorted_list_iter:
                sorted_data_to_plot[agent][action][num_iter] = action_data

    data_to_plot = sorted_data_to_plot
    #print("data_to_plot: ", data_to_plot)


    # extract important data
    config = round_dict(config, num=2)
    max_progress_in_env_0 = get_min_time_to_complete_modified(curr_state=results['trajectory_0'][0]+results['trajectory_1'][0], goal_state=goal_state, config=config, agent= 0)
    max_progress_in_env_1 = get_min_time_to_complete_modified(curr_state=results['trajectory_0'][0]+results['trajectory_1'][0], goal_state=goal_state, config=config, agent= 1)

    exp_data_to_print = str("Yaw rates agent 0: " + str(config["ang_velocity_0"]) + "\n" + "Yaw rates agent 1: " + str(config["ang_velocity_1"]) + "\n" + 
                            "Velocities agent 0: " + str(config["velocity_0"]) + "\n" +"Velocities agent 1: " + str(config["velocity_1"]) + "\n" + 
                        "Timesteps to reach terminal state agent 0: " + str(max_progress_in_env_0) + "\n" + "Timesteps to reach terminal state agent 1: " + str(max_progress_in_env_1) + "\n" +
                    "Game length (max. timesteps): " + str(game_length))

    # Add text to the right of the figure
    #ax2.text(-1.0, 0.1, exp_data_to_print, transform=ax1.transAxes, fontsize=9, verticalalignment='top')

    # PLOT DATA
    #fig, (ax2, ax1) = plt.subplots(1,2, figsize=(12, 6))

    # Create a twin Axes object
    ax11 = ax1.twinx()

    # Define color space
    num_actions_0 = len(data_to_plot[0].items())
    num_actions_1 = len(data_to_plot[1].items())
    orange_colors = plt.cm.Oranges(np.linspace(0.2, 0.8, num_actions_0))
    blue_colors = plt.cm.Blues(np.linspace(0.2, 0.8, num_actions_1))

    agent_color = [orange_colors, blue_colors]
    plotted_colors = [[],[]]

    for agent, action_dict in data_to_plot.items():
        i = 0
        for action, data_num_iter in action_dict.items():
            x_iter = []
            y_mean_runtime = []
            y_mean_payoff = []
            y_var_payoff = []
            for num_iter, data_point in data_num_iter.items():
                x_iter.append(num_iter)
                y_mean_runtime.append(data_point[0])
                y_mean_payoff.append(data_point[1])
                y_var_payoff.append(data_point[2])
            ax1.plot(x_iter, y_mean_payoff, color=agent_color[agent][i])
            # Use the function to filter your data
            x_iter_realistic, y_mean_runtime_realistic = filter_data(x_iter, y_mean_runtime, window_size=3, threshold=1000)
            # Plot the filtered data
            ax11.plot(x_iter_realistic, y_mean_runtime_realistic, color="black", linestyle='--', label='Runtime', linewidth=1)
            #ax11.plot(x_iter, y_mean_runtime, color="black", linestyle='--', label='Runtime', linewidth=1)
            plotted_colors[agent].append(agent_color[agent][i])
            i += 1


    # Adding label iter_krit=2030 on the right side
    #finding n_iter where variance of the mean_payoffs of all actions is below a certain threshold
    """iter_krit = 0
    for action, data_points in data_to_plot[str(0)].items():
        for data_point in data_points:
            if data_point[-1] < var_threshold:
                iter_krit = data_point[0]
                break
        if iter_krit != 0:
            break
    ax1.axvline(x=iter_krit, color='black', linestyle='--')"""

    # Adding legend
    median_index_0 = int(statistics.median(range(len(plotted_colors[0]))))
    median_index_1 = int(statistics.median(range(len(plotted_colors[1]))))

    # PLOT map
    plot_config_map(ax2, config, result_dict= result_dict_example, traj_timesteps = 1, env_timestep=0, finish_line=13, main_agent=0, game_length=game_length)
    ax2.title.set_text('Overtaking scenario with game length: {}'.format(game_length))

    # change labels and scales
    ax1.set_xticks(np.arange(100, 5000, step=500))

    # Add legend below the figure
    ax1.legend([plt.Line2D([0], [0], color=plotted_colors[0][median_index_0], lw=2),
                plt.Line2D([0], [0], color=plotted_colors[1][median_index_1], lw=2),
                plt.Line2D([0], [0], color='black', linestyle='--', lw=2)],
                ['Actions of Agent 0', 'Actions of Agent 1', 'Runtime'],
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                fancybox=True,
                shadow=True,
                ncol=3)

    # Adding labels and title
    ax1.set_xlabel(r'$n_{iter}$')
    ax1.set_ylabel('$\overline{Q}$($n_{iter}$)')
    ax1.set_title(r'$\overline{Q}$-values for overtaking scenario')

    # Plot the runtime data
    ax11.set_ylabel('Runtime (in sec)')

# filter out too high values
def filter_data(x, y, window_size=10, threshold=1000):
    # Convert to numpy arrays for easier manipulation
    x_np = np.array(x)
    y_np = np.array(y)

    # Calculate the moving average and standard deviation
    y_avg = uniform_filter1d(y_np, size=window_size)
    y_std_ges = np.std(y_np)
    y_std = np.std(y_avg)
    print("y_std_ges: ", y_std_ges)
    print("y_std: ", y_std)

    # Identify values within an order of magnitude of the moving average
    realistic_indices = y_np < threshold

    # Filter out unrealistic high values
    x_realistic = x_np[realistic_indices]
    y_realistic = y_np[realistic_indices]

    # Perform a linear regression
    model = LinearRegression()
    model.fit(x_realistic.reshape(-1, 1), y_realistic)

    return x_realistic, y_realistic

if __name__ == "__main__":
    # Global plot configuration
    exp_name = "01_convergence_overtaking_single_mcts"
    data_dir = '/home/enjay/0_thesis/01_MCTS/data/'
    input_dir = os.path.join(data_dir, exp_name, "raw")
    result_dir = '/home/enjay/0_thesis/01_MCTS/results/'


    num_game_length = 15 # plot game length from 1 to n
    n_cols = 3
    n_rows = int(num_game_length/n_cols)
    #n_rows = int(np.sqrt(num_game_length))
    #n_cols = 2*int(np.sqrt(num_game_length))
    print("n_rows: ", n_rows)
    print("n_cols: ", n_cols)

    # threshhold for visualizing reduced variance
    var_threshold = 0.005

    goal_state = [14,3] #[x,y]

    # Add code to generate subfigure content here
    fig, axs = plt.subplots(n_rows, 2*n_cols, figsize=(8*n_cols, 6*n_rows))

    i = 1
    for ix_row in range(0,n_rows):
        for ix_col in range(0, 2*n_cols, 2):
            plot_figure(axs[ix_row, ix_col+1], axs[ix_row, ix_col], input_dir, game_length=i)
            i += 1

    title_text = "Title for Axes {} and {}".format(ix_col+1, ix_col+2)
    fig.suptitle(title_text, fontsize=12, fontweight='bold')
    
    # Save the plot
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust the vertical space between subplots
    plt.savefig(os.path.join(result_dir,exp_name,'table_plot.png'), dpi=200)

    