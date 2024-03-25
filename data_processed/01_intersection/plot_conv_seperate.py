import sys
import os
import pandas as pd
import ast
from matplotlib.lines import Line2D

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LinearRegression
import re
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter



import statistics
import numpy as np
import statistics


from plot_data_utilities import *
from plot_data_utilities import plot_config_map

from process_data_utilities import *
from environment_utilities import *




"""import matplotlib as mpl
mpl.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import tikzplotlib"""

import matplotlib.pyplot as plt


#############
# Compute alpha_t if it wasn't saved in the results


# Define a function that formats the tick values
def format_tick(value, tick_number):
    return f'{value * 1e-6}'

# Create a formatter
formatter = FuncFormatter(format_tick)

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

def plot_map(ax, path_curr_experiment, game_length):
    config = None
    result_dict = None
    for subfolder_iter in os.listdir(path_curr_experiment):
        subfolder_iter_path = os.path.join(path_curr_experiment, subfolder_iter)

        if read_json_entry(directory=subfolder_iter_path, filename="results.json", entry="T_terminal") == 2: # fix game length for visualization here
            if os.path.isdir(subfolder_iter_path):
                config = read_json_file(directory=subfolder_iter_path, filename="config.json")
                
            
                for subfolder_run in os.listdir(subfolder_iter_path):
                        subfolder_run_path = os.path.join(subfolder_iter_path, subfolder_run)

                        if os.path.isdir(subfolder_run_path):
                            # add alpha_t to the list
                            result_dict = read_json_file(directory=subfolder_run_path, filename="results.json")
                            break
                break

    # PLOT map
    plot_config_map(ax, config, result_dict, traj_timesteps = 1, env_timestep=0, finish_line=13, main_agent=0, game_length=game_length)
    #ax.title.set_text('Reachable states within the environment')

def plot_q_values(ax1, data_frame, styles):
    ax1.set_facecolor('white')

    # Group the DataFrame by 'agent' and 'action'
    grouped = data_frame.groupby(['agent', 'action'])

    # Plot payoff_mean data
    for (agent, action), group in grouped:
        ax1.plot(group['num_iter'], group['payoff_mean'], label=f'Agent {agent}, Action {action}', **styles[f'{agent}: {action}'])

    # Change labels and scales
    ax1.set_xticks(np.arange(100, 5000, step=500))

    # Adding labels and title
    ax1.set_xlabel(r'$n_{iter}$')
    ax1.set_ylabel('$\overline{Q}$($n_{iter}$)')

    # Plot the runtime data
    ax11 = ax1.twinx()
    for (agent, action), group in grouped:
        ax11.plot(group['num_iter'], group['runtime'], label='runtime', **styles['runtime'])
        ax11.set_ylabel('Runtime (in sec)')
        break

"""   
def plot_q_values(ax1, path_curr_data, game_length, weight_interm, styles):
    ax1.set_facecolor('white')
    # import variance data from csv
    df = pd.read_csv(os.path.join(path_curr_data,"gamelength_{}-weightinterm_{}.csv".format(game_length, weight_interm)))

    # Iterate over all rows in the DataFrame
    for agent, group in df.groupby(['agent']):
        agent = list(agent)[0]
        i = 0
        for action, group_action in group.groupby(['action']):
            action = [round(val, 2) for val in ast.literal_eval(list(action)[0])]
            ax1.plot(group_action['num_iter'], group_action['payoff_mean'], label=f'Agent {agent}: {action}', **styles[f'{agent}: {action}'])
            i += 1

    # change labels and scales
    ax1.set_xticks(np.arange(100, 5000, step=500))

    # Adding labels and title
    ax1.set_xlabel(r'$n_{iter}$')
    ax1.set_ylabel('$\overline{Q}$($n_{iter}$)')

    # Plot the runtime data
    ax11 = ax1.twinx()
    for agent, group in df.groupby(['agent']):
        agent = list(agent)[0]
        i = 0
        for action, group_action in group.groupby(['action']):
            action = [round(val, 2) for val in ast.literal_eval(list(action)[0])]
            ax11.plot(group_action['num_iter'], group_action['runtime'], label='runtime', **styles['runtime'])
            ax11.set_ylabel('Runtime (in sec)')
            break
"""

def plot_variance(ax, data_frame, styles):
    ax.set_facecolor('white')

    # Group the DataFrame by 'agent' and 'action'
    grouped = data_frame.groupby(['agent', 'action'])

    # Plot variance data
    for (agent, action), group in grouped:
        ax.plot(group['num_iter'], group['payoff_var'], label=f'Agent {agent}, Action {action}', **styles[f'{agent}: {action}'])

    # Change labels and scales
    ax.set_yscale('log')
    ax.set_xticks(np.arange(100, 5000, step=500))

    # Adding labels
    ax.set_xlabel(r'$n_{iter}$')
    ax.set_ylabel('Var(Q)')

"""def plot_variance(ax, path_curr_data, game_length, weight_interm, styles):
    ax.set_facecolor('white')
    # import variance data from csv
    df = pd.read_csv(os.path.join(path_curr_data,"gamelength_{}-weightinterm_{}.csv".format(game_length, weight_interm)))
    

    for agent, group in df.groupby(['agent']):
        agent = list(agent)[0]
        i = 0
        for action, group_action in group.groupby(['action']):
            action = [round(val, 2) for val in ast.literal_eval(list(action)[0])]
            ax.plot(group_action['num_iter'], group_action['payoff_var'], label=f'{agent}: {action}', **styles[f'{agent}: {action}'])
            i += 1

    # change labels and scales
    ax.set_yscale('log')
    ax.set_xticks(np.arange(100, 5000, step=500))
    ax.set_xlabel(r'$n_{iter}$')
    ax.set_ylabel('Var(Q)')
    #ax.set_title('Variance of Q-values')"""


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

def plot_legend(styles, result_dir=None):
    # Plot the legend
    i = 0
    for style in styles:
        fig, ax = plt.subplots(figsize=(5, 5))
        # Create custom lines that will be used as legend entries
        custom_lines = [Line2D([0], [0], **style_element) for label, style_element in style.items()]
        ax.axis('off')
        ax.legend(custom_lines, style.keys(), loc='center')
        plt.tight_layout()
        plt.savefig(result_dir+"/legend_style_{}.svg".format(i), format="svg", bbox_inches='tight')
        plt.savefig(result_dir+"/legend_style_{}.png".format(i), dpi=200, bbox_inches='tight')
        i += 1 # different style part
    #ax.set_title('Actions [v, $\omega$] \nof both agents')

if __name__ == "__main__":
    #exp_name = "00_convergence_following/duct-sampling_informed-expand_every"
    exp_name = "00_convergence_overtaking/duct-sampling_informed-expand_every"

    data_dir = os.path.dirname(__file__)
    main_dir = os.path.abspath(os.path.join(data_dir, os.pardir))
    input_dir_raw = os.path.join(main_dir, "data_raw", exp_name)
    input_dir_processed = os.path.join(data_dir, exp_name)

    result_dir = os.path.join(main_dir, "results/")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    num_game_length = 15 # plot game length from 1 to n

    # threshhold for visualizing reduced variance

    goal_state = [14,3] #[x,y]

    # define styles
    # Define color space
    num_actions_0 = 9
    num_actions_1 = 9
    orange_colors = plt.cm.Oranges(np.linspace(0.2, 0.8, num_actions_0))
    blue_colors = plt.cm.Blues(np.linspace(0.2, 0.8, num_actions_1))
    markers = ['o', 'v', '^', '*', '+', 'x', 'D', 'd', '|', '_']
    
    styles_0 = {
        '0: [0.0, -1.57]': {'color': orange_colors[0], 'marker': '<'},
        '0: [0.0, 0.0]': {'color': orange_colors[1], 'marker': 'o'},
        '0: [0.0, 1.57]': {'color': orange_colors[2], 'marker': '>'},
        '0: [1.0, -1.57]': {'color': orange_colors[3], 'marker': '<'},
        '0: [1.0, 0.0]': {'color': orange_colors[4], 'marker': '^'},
        '0: [1.0, 1.57]': {'color': orange_colors[5], 'marker': '>'},
        '0: [2.0, -1.57]': {'color': orange_colors[6], 'marker': '<'},
        '0: [2.0, 0.0]': {'color': orange_colors[7], 'marker': '^'},
        '0: [2.0, 1.57]': {'color': orange_colors[8], 'marker': '>'},
    }
    styles_1 = {
        '1: [0.0, -1.57]': {'color': blue_colors[0], 'marker': '<'},
        '1: [0.0, 0.0]': {'color': blue_colors[1], 'marker': 'o'},
        '1: [0.0, 1.57]': {'color': blue_colors[2], 'marker': '>'},
        '1: [1.0, -1.57]': {'color': blue_colors[3], 'marker': '<'},
        '1: [1.0, 0.0]': {'color': blue_colors[4], 'marker': '^'},
        '1: [1.0, 1.57]': {'color': blue_colors[5], 'marker': '>'},
        '1: [2.0, -1.57]': {'color': orange_colors[6], 'marker': '<'},
        '1: [2.0, 0.0]': {'color': orange_colors[7], 'marker': '^'},
        '1: [2.0, 1.57]': {'color': orange_colors[8], 'marker': '>'},
    }
    styles_runtime = {
        'runtime': {'color': 'black', 'linestyle': '--'},
    }

    styles = {**styles_0, **styles_1, **styles_runtime}

    """styles = {
        '0: [0.0, -1.57]': {'color': orange_colors[0], 'marker': '<'},
        '0: [0.0, 0.0]': {'color': orange_colors[1], 'marker': 'o'},
        '0: [0.0, 1.57]': {'color': orange_colors[2], 'marker': '>'},
        '0: [1.0, -1.57]': {'color': orange_colors[3], 'marker': '<'},
        '0: [1.0, 0.0]': {'color': orange_colors[4], 'marker': '^'},
        '0: [1.0, 1.57]': {'color': orange_colors[5], 'marker': '>'},
        '0: [2.0, -1.57]': {'color': orange_colors[6], 'marker': '<'},
        '0: [2.0, 0.0]': {'color': orange_colors[7], 'marker': '^'},
        '0: [2.0, 1.57]': {'color': orange_colors[8], 'marker': '>'},
        '1: [0.0, -1.57]': {'color': blue_colors[0], 'marker': '<'},
        '1: [0.0, 0.0]': {'color': blue_colors[1], 'marker': 'o'},
        '1: [0.0, 1.57]': {'color': blue_colors[2], 'marker': '>'},
        '1: [1.0, -1.57]': {'color': blue_colors[3], 'marker': '<'},
        '1: [1.0, 0.0]': {'color': blue_colors[4], 'marker': '^'},
        '1: [1.0, 1.57]': {'color': blue_colors[5], 'marker': '>'},
        'runtime': {'color': 'black', 'linestyle': '--'},
        # Add more labels and styles as needed
        }
    """


    """for game_length in range(1, num_game_length+1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 4, width_ratios=[1.5, 1, 1, 0.3])  # Creates a grid with 1 row and 4 columns

        axs = [plt.subplot(gs[i]) for i in range(4)]  # Creates 4 subplots
        
        plt.subplots_adjust(wspace=0.5)  # Adjust this value as needed


        plot_map(axs[0], input_dir_raw, game_length)
        plot_q_values(axs[1], input_dir_raw, game_length, styles)
        plot_variance(axs[2], input_dir_processed, game_length, styles)
        plot_legend(axs[3], styles)
        title_text = "Convergence of MCTS in overtaking scenario for a game lengths of  {}".format(game_length)
        fig.suptitle(title_text, fontsize=12, fontweight='bold')
        
        # Save the plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=2)  # Adjust the vertical space between subplots
        try:
            plt.savefig(result_dir+exp_name+"/{}.svg".format(game_length), format="svg")
            plt.savefig(result_dir+exp_name+"/{}.png".format(game_length), dpi=200)
            #tikzplotlib.save(result_dir+exp_name+"/{}.tex".format(game_length),standalone=True)
        except:
            plt.savefig(result_dir+exp_name+"/{}.svg".format(game_length), format="svg")
            plt.savefig(result_dir+exp_name+"/{}.png".format(game_length), dpi=200)
        plt.close()"""
    
    for game_length in range(1, num_game_length+1):
        plot_functions = [plot_map, plot_q_values, plot_variance]
        plot_args = [(input_dir_raw, game_length), (input_dir_processed, game_length, styles), (input_dir_processed, game_length, styles)]
        plot_names = ['Reachable_states_within_the_environment', 'Q-values_averaged_over_10_runs', 'Variance_of_Q-values']
        fig_sizes = [(10, 5), (6, 5), (6, 5)]


        for plot_func, plot_arg, plot_name, fig_size in zip(plot_functions, plot_args, plot_names, fig_sizes):
            fig, ax = plt.subplots(figsize=fig_size)
            plot_func(ax, *plot_arg)
            #title_text = "Convergence of MCTS in overtaking scenario for a game lengths of  {} - {}".format(game_length, plot_name)
            #fig.suptitle(title_text, fontsize=12, fontweight='bold')

            # Save the plot
            plt.tight_layout()

            save_dir = result_dir+exp_name
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(result_dir+exp_name+"/{}_{}.svg".format(game_length, plot_name), format="svg")
            plt.savefig(result_dir+exp_name+"/{}_{}.png".format(game_length, plot_name), dpi=200)
            #tikzplotlib.save(result_dir+exp_name+"/{}_{}.tex".format(game_length, plot_name),standalone=True)
            plt.close()
        print("Saved plot for game length: ", game_length)
        




    