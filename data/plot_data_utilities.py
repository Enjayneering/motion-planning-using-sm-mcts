import os
import sys
import numpy as np
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
from process_data_utilities import *
sys.path.append('/home/enjay/0_thesis/01_MCTS/src')
from utilities.env_utilities import *


def plot_config_map(ax, config, result_dict= None, traj_timesteps = None, env_timestep=None, finish_line=None, main_agent=0, game_length=None):
    colormap = {'red': (192/255, 67/255, 11/255), 
                'darkred': (83/255, 29/255, 0/255),
                'blue': (78/255, 127/255, 141/255), 
                'darkblue': (38/255, 57/255, 63/255),
                'yellow': (218/255, 181/255, 100/255), 
                'grey': (71/255, 63/255, 61/255)}
    
    if env_timestep is None:
        env_timestep = result_dict['T_terminal']

    # plotting the trajectory of two agents on a 2D-plane and connecting states with a line and labeling states with timestep | trajectory: list of states [x1, y1, x2, y2, timestep]
    if traj_timesteps is None:
        traj_timesteps = result_dict['T_terminal']+1
    trajectory_0 = result_dict['trajectory_0'][:traj_timesteps]
    trajectory_1 = result_dict['trajectory_1'][:traj_timesteps]
    finish_line = finish_line
    
    # define plotting environment
    xmax = max((line.count("#")+line.count(".")) for line in config["env_def"][str(env_timestep)].split('\n')) - 1
    ymax = config["env_def"][str(env_timestep)].count('\n') - 1
    #print("xmax: {}, ymax: {}".format(xmax, ymax))
    ax.set_xlim([-0.5, xmax+1.5])
    ax.set_ylim([-0.5, ymax+1.5][::-1]) # invert y-axis to fit to the environment defined in the numpy array


    # turn of the outter axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.autoscale(tight=True)


    # Plot the grid map 
    plot_map(ax, array=get_current_grid(config["env_def"], env_timestep), facecolor=colormap['yellow'], edgecolor=colormap['grey'])

    # Visualize alpha_t effect on reachability
    plot_reachable_points(ax, array=get_current_grid(config["env_def"], env_timestep), config=config, game_length=game_length, color_0=colormap['red'], color_1=colormap['blue'])
    
    # Plot coordinate system
    plotCOS(ax, x_orig=0, y_orig=0, scale=1, colormap=colormap)
                 
    # plot finish line
    #plot_finish_line(ax, finish_line=finish_line, ymax=ymax)
    
    # plot trajectories
    if main_agent == 0:
        plot_trajectory(ax, 0, trajectory_0, facecolor=colormap['red'], edgecolor=colormap['darkred'] ,label='Agent 0 (Us)', zorder=100, alpha=1)
        plot_trajectory(ax, 1, trajectory_1, facecolor=colormap['blue'], edgecolor=colormap['darkblue'] , label='Agent 1 (Opponent)', zorder=50, alpha=1)
    elif main_agent == 1:
        plot_trajectory(ax, 0, trajectory_0, facecolor=colormap['red'], edgecolor=colormap['darkred'] ,label='Agent 0 (Us)', zorder=50, alpha=1)
        plot_trajectory(ax, 1, trajectory_1, facecolor=colormap['blue'], edgecolor=colormap['darkblue'] , label='Agent 1 (Opponent)', zorder=100, alpha=1) 
    
    
    # plot legend
    #ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, fancybox=True, framealpha=0.5, bbox_transform=fig.transFigure) #title='Trajectory of'
    ax.set_aspect('equal', 'box')
    
    # remove x and y labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

def get_reachable_states(init_state, vel_list, yaw_rate_list, grid, game_length, dt=1):
    visited = np.zeros(grid.T.shape, dtype=bool)
    queue = [(init_state[0], init_state[1], init_state[2], 0)]  # Queue of states to visit (x, y, yaw)
    states_visited = []
    while queue and queue[0][3] <= game_length:
        x, y, yaw, timestep = queue.pop(0)  # Get the next state to visit

        # Mark the current state as visited
        if timestep <= game_length:
            visited[int(x), int(y)] = True

        # Apply the maximum actions to get the next state
        for vel in vel_list:
            for yaw_rate in yaw_rate_list:
                x_next = x + vel * np.cos(yaw) * dt
                y_next = y + vel * np.sin(yaw) * dt
                yaw_next = (yaw + yaw_rate * dt)# % (2 * np.pi)
                timestep_next = timestep + dt

                # If the next state is within the grid and hasn't been visited yet, add it to the queue
                if 0 <= x_next < visited.shape[0] and 0 <= y_next < visited.shape[1] and timestep_next <= game_length and (x_next, y_next, yaw_next) not in states_visited:
                    queue.append((x_next, y_next, yaw_next, timestep_next))
                    states_visited.append((x_next, y_next, yaw_next))
    # Assuming arr is your n x m numpy array
    visited_flipped = np.flip(visited, axis=1)

    visited_or = np.logical_or(visited, visited_flipped)

    return visited_or

def plot_reachable_points(ax, array, config, game_length=None, color_0='red', color_1='blue'):
    init_state = get_init_state(config["env_raceconfig"]) # [x0, y0, theta0, x1, y1, theta1]
    reachable_states_0 = get_reachable_states([init_state[0], init_state[1], init_state[2]], config["velocity_0"], config["ang_velocity_0"], array, game_length, dt=1)
    reachable_states_1 = get_reachable_states([init_state[3], init_state[4], init_state[5]], config["velocity_1"], config["ang_velocity_1"], array, game_length, dt=1)
    
    for row in range(reachable_states_0.shape[0]):
        for col in range(reachable_states_0.shape[1]):
            # agent 0 color
            if reachable_states_0[row, col] == True:
                circle = Circle((row, col), radius=0.4, facecolor='none', edgecolor=color_0, zorder=2)
                ax.add_patch(circle)
    for row in range(reachable_states_1.shape[0]):
        for col in range(reachable_states_1.shape[1]):
            # agent 1 color
            if reachable_states_1[row, col] == True:
                circle = Circle((row, col), radius=0.2, facecolor=color_1, edgecolor=color_1, zorder=2)
                ax.add_patch(circle)

def is_reachable_point(init_state, max_vel, gridpoint, game_length):
    x_init = init_state[0]
    y_init = init_state[1]
    if manhattan_distance(x_init, y_init, gridpoint[0], gridpoint[1]) <= max_vel * game_length:
        return True
    else:
        return False
    


def plot_single_run(Game, path_savefig, result_dict= None, timestep=None, main_agent=0):
    colormap = {'red': (192/255, 67/255, 11/255), 
                'darkred': (83/255, 29/255, 0/255),
                'blue': (78/255, 127/255, 141/255), 
                'darkblue': (38/255, 57/255, 63/255),
                'yellow': (218/255, 181/255, 100/255), 
                'grey': (71/255, 63/255, 61/255)}
    
    if timestep is None:
        timestep = result_dict['T_terminal']

    # plotting the trajectory of two agents on a 2D-plane and connecting states with a line and labeling states with timestep | trajectory: list of states [x1, y1, x2, y2, timestep]
    trajectory_0 = result_dict['trajectory_0']
    trajectory_1 = result_dict['trajectory_1']
    finish_line = Game.env.finish_line
    
    # define plotting environment
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.9)
    xmax = max((line.count("#")+line.count(".")) for line in Game.config.env_def[0].split('\n')) - 1
    ymax = Game.config.env_def[0].count('\n') - 1
    #print("xmax: {}, ymax: {}".format(xmax, ymax))
    ax.set_xlim([-0.5, xmax+1.5])
    ax.set_ylim([-0.5, ymax+1.5][::-1]) # invert y-axis to fit to the environment defined in the numpy array
    


    # turn of the outter axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # create a frame around the figure
    #frame = plt.gca()
    #frame.set_frame_on(True)
    #frame.patch.set_edgecolor(grey_color)
    #frame.patch.set_linewidth(5)  # Increase the linewidth to make the frame thicker
    #ax.set_title("Trajectory")


    # Plot the grid map 
    last_map = get_current_grid(Game.config.env_def, timestep)
    #color_range = np.linspace(218/255, 100/255, len(grid_maps))
    #gridcolor = [(218/255, 181/255, color_range[i]) for i in range(len(grid_maps))]
    #for i, grid_map in enumerate(grid_maps):
    
    plot_map(ax, array=last_map, facecolor=colormap['yellow'], edgecolor=colormap['grey'])

    # Plot coordinate system
    plotCOS(ax, x_orig=0, y_orig=0, scale=0.8, colormap=colormap)
                 
    # plot finish line
    #TODO: plot finish line?
    #plot_finish_line(ax, finish_line=finish_line, ymax=ymax)

    #plot_goal_states(ax, goal_state=Game.env.goal_state, agent=0, color=colormap['red'])
    #plot_goal_states(ax, goal_state=Game.env.goal_state, agent=1, color=colormap['blue'])
    plot_centerline(ax, centerlines=Game.env.centerlines, agent=0, color=colormap['red'], alpha=0.5)
    plot_centerline(ax, centerlines=Game.env.centerlines, agent=1, color=colormap['blue'], alpha=0.5)
    
    # plot trajectories
    if main_agent == 0:
        plot_trajectory(ax, 0, trajectory_0, facecolor=colormap['red'], edgecolor=colormap['darkred'] ,label='Agent 0 (Us)', zorder=100, alpha=1)
        plot_trajectory(ax, 1, trajectory_1, facecolor=colormap['blue'], edgecolor=colormap['darkblue'] , label='Agent 1 (Opponent)', zorder=50, alpha=1)
    elif main_agent == 1:
        plot_trajectory(ax, 0, trajectory_0, facecolor=colormap['red'], edgecolor=colormap['darkred'] ,label='Agent 0 (Us)', zorder=50, alpha=1)
        plot_trajectory(ax, 1, trajectory_1, facecolor=colormap['blue'], edgecolor=colormap['darkblue'] , label='Agent 1 (Opponent)', zorder=100, alpha=1) 
    
    
    # plot legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, fancybox=True, framealpha=0.5, bbox_transform=fig.transFigure) #title='Trajectory of'
    ax.set_aspect('equal', 'box')
    # remove x and y labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.axis('off')
    fig.tight_layout()

    # save figure
    fig.savefig(os.path.join(path_savefig, "trajectory{}_{}.png".format(main_agent, timestep)), dpi=300, bbox_inches='tight')
    # Remove empty borders
    plt.autoscale(tight=True)
    plt.close(fig)

def plot_finish_line(ax, finish_line=None, ymax=None):
    if finish_line is not None:
        # Replace dotted line with periodic square patches
        patch_width = 0.5
        patch_height = ymax

        i = 0
        while i * patch_width < ymax:
            if i % 2 == 0:
                rect = Rectangle((finish_line - patch_width, i * patch_width), patch_width, patch_width, facecolor='lightgrey', edgecolor='lightgrey')
            else:
                rect = Rectangle((finish_line, i * patch_width), patch_width, patch_width, facecolor='lightgrey', edgecolor='lightgrey')
            ax.add_patch(rect)
            i += 1
    else:
        pass

def plot_goal_states(ax, goal_state, agent=0, color='black'):
    if goal_state is not None:
        x_goal = goal_state[f'x{agent}']
        y_goal = goal_state[f'y{agent}']
        ax.plot(x_goal, y_goal, 'o', color=color, markersize=5, label=f'Goal agent {agent}')
    else:
        pass

def plot_centerline(ax, centerlines, agent=0, color='black', alpha=1):
    x_centerline = [c[0] for c in centerlines[agent]]
    y_centerline = [c[1] for c in centerlines[agent]]
    ax.plot(x_centerline, y_centerline, "x--", color=color, alpha=alpha, label=f'Centerline Agent {agent}', zorder=1)

def plot_trajectory(ax, agent, trajectory, linewidth=4, facecolor=None, edgecolor=None, label=None, fontsize=4, zorder=None, alpha=None):
    xy_visited = []
    x_values = [sublist[0] for sublist in trajectory]
    y_values = [sublist[1] for sublist in trajectory]
    theta_values = [sublist[2] for sublist in trajectory]
    timesteps = [sublist[3] for sublist in trajectory]
    # plot connecting lines
    ax.plot(x_values, y_values, color=facecolor, linestyle='-', linewidth=linewidth, label=label, zorder=zorder, alpha=alpha*0.75)

    # annotate timesteps
    for i in range(len(trajectory)):
        annotate_state(ax, x_values[i], y_values[i], theta_values[i], timesteps[i], agent, visit_count=xy_visited.count([x_values[i], y_values[i]]), facecolor=facecolor, edgecolor=edgecolor, fontsize=fontsize, zorder=zorder+1, alpha=alpha)
        xy_visited.append([x_values[i], y_values[i]])

def annotate_state(ax, x, y, theta, timestep, agent, annotate=False, visit_count=0, facecolor=None, edgecolor=None, fontsize=None, zorder=None, alpha=None, arrowscale=2):
    # plot orientations as arrows
    arrow_length = 0.2+0.2*(visit_count)*arrowscale
    arrow_width = 0.2*arrowscale
    head_width = 0.4*arrowscale
    head_length = 0.4*arrowscale
    linewidth = 0.4*arrowscale
    annot_alignment = 0.25

    arrow_dx = arrow_length * np.cos(theta)
    arrow_dy = arrow_length * np.sin(theta)
    ax.arrow(x, y, arrow_dx, arrow_dy, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, width=arrow_width, head_width=head_width , head_length=head_length, zorder=zorder+visit_count, alpha=alpha)
    
    # annotate timestep on arrow
    if annotate == True:
        ax.annotate(timestep, (x+arrow_dx+annot_alignment*head_length*np.cos(theta), y+arrow_dy+annot_alignment*head_length*np.sin(theta)), color=edgecolor, fontsize=fontsize*arrowscale, textcoords="offset points", xytext=(0, 0), ha='center', va='center', zorder=zorder+visit_count, alpha=alpha)
    else:
        # annotate agent instead of timestep
        ax.annotate(agent, (x+arrow_dx+annot_alignment*head_length*np.cos(theta), y+arrow_dy+annot_alignment*head_length*np.sin(theta)), color=edgecolor, fontsize=fontsize*arrowscale, textcoords="offset points", xytext=(0, 0), ha='center', va='center', zorder=zorder+visit_count, alpha=alpha, fontweight='bold')

    # add cicrle in the middle of the state
    loc_circle = plt.Circle((x, y), radius=0.1*arrowscale, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.4, zorder=zorder+visit_count, alpha=alpha)
    ax.add_patch(loc_circle)

def plotCOS(ax, x_orig, y_orig, scale, colormap, fontsize = 8, alpha=1):
    thickness = 0.2
    ax.arrow(x_orig, y_orig, scale, 0, head_width=thickness, head_length=thickness, fc=colormap['grey'], ec=colormap['grey'], alpha=alpha, zorder=4)
    ax.arrow(x_orig, y_orig, 0, scale, head_width=thickness, head_length=thickness, fc=colormap['grey'], ec=colormap['grey'], alpha=alpha, zorder=4)
    ax.plot(x_orig, y_orig, 'o', color=colormap['grey'], markersize=5, alpha=alpha, zorder=5)

    ax.text(x_orig, y_orig-0.3, "(0,0)", fontsize=fontsize, ha='center', va='center', alpha=alpha, zorder=4)
    ax.text(x_orig+scale, y_orig-0.3, "x", fontsize=fontsize, ha='center', va='center', alpha=alpha, zorder=4)
    ax.text(x_orig-0.3, y_orig+scale, "y", fontsize=fontsize, ha='center', va='center', alpha=alpha, zorder=4)

def plot_map(ax, array, facecolor=None, edgecolor=None, timestep_max=None, roadcolor='floralwhite'):
    # plot the racing gridmap
    for row in range(len(array)):
        for col in range(len(array[0])):
            if array[row][col] == 1: # static obstacles
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, facecolor=facecolor, edgecolor=edgecolor, alpha=1, zorder=3))
            if array[row][col] == 2: # dynamic obstacles
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, facecolor=facecolor, edgecolor=edgecolor, alpha=1, hatch="//", zorder=3))
            elif array[row][col] == 0: # free space, road
                ax.plot(col, row, 'o', color='dimgrey', markersize=1, alpha=1)
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, color=roadcolor, linewidth=0, alpha=1, zorder=1))
    # Add an edge around the whole array
    ax.add_patch(Rectangle((-0.5, -0.5), len(array[0]), len(array), fill=None, edgecolor=edgecolor, linewidth=2))

def get_current_grid(grid_dict, timestep):
    for grid_timeindex in reversed(grid_dict.keys()):
        if int(grid_timeindex) <= timestep:
            current_grid = grid_dict[grid_timeindex]
            break
    occupancy_grid_define = current_grid.replace('.', '0').replace('0', '0').replace('x', '0').replace('1', '0').replace('#', '1').replace('+', '2').replace('F', '0')
    lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
    transformed_grid = [list(map(int, line)) for line in lines]
    occupancy_grid = np.array(transformed_grid)
    return occupancy_grid


def is_terminal(Game, state_obj, max_timestep=None):
        # terminal condition
        #print(state_obj.timestep, Game.config.max_timehorizon)
        #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
        """if Game.env.finish_line is not None:
            finish_line = Game.env.finish_line
            if state_obj.x0 >= finish_line or state_obj.x1 >= finish_line:
                #print("Terminal state_obj reached")
                return True
        elif Game.env.centerlines is not None:
            if agent_has_finished(Game, state_obj, agent=0) or agent_has_finished(Game, state_obj, agent=1):
                #print("Terminal state_obj reached")
                #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
                return True"""
        if state_obj.timestep > max_timestep:
            #print("Max timehorizon reached: {}".format(state_obj.timestep))
            return True
        else:
            return False

def get_max_timehorizon(Game):
    min_time = get_min_time_to_complete(Game)

    max_game_timehorizon = int(Game.config.alpha_terminal * min_time)
    #print("Max game timehorizon: {}".format(max_game_timehorizon))
    return max_game_timehorizon

def get_min_time_to_complete(Game, curr_state=None):
    # state: [x0, y0, theta0, x1, y1, theta1, time]
    min_times = []
    
    if curr_state is None:
            curr_state = Game.init_state

    final_state = Game.terminal_state
    dist_0 = distance(curr_state[0:2], final_state[0:2])
    dist_1 = distance(curr_state[3:5], final_state[3:5])
    #dist_1 = distance(curr_state[3:5], final_state[3:5])
    max_velocity_0 = np.max(Game.config["velocity_0"])
    max_velocity_1 = np.max(Game.config["velocity_1"])
    min_times.append(dist_0/max_velocity_0)
    min_times.append(dist_1/max_velocity_1)    
    return max(min_times)

def coll_count(joint_trajectory):
    coll_count = 0
    for t in range(len(joint_trajectory)-1):
        line_points_0 = np.linspace(joint_trajectory[t][0:2], joint_trajectory[t+1][0:2], num=10).tolist()
        line_points_1 = np.linspace(joint_trajectory[t][3:5], joint_trajectory[t+1][3:5], num=10).tolist()
        if any(distance(point_0, point_1) <= 0.5 for point_0 in line_points_0 for point_1 in line_points_1):
            coll_count += 1
    return coll_count

def agent_has_finished(Game, state_obj, agent=0):
    max_progress = Game.env.centerlines[agent][-1][-1] # TODO: specify goal
    if find_closest_waypoint(Game.env.centerlines[agent], state_obj.get_state(agent=agent))[-1] >= max_progress:
        return True
    else:
        return False
    
def get_winner(Game, state):
    # TODO: adjust for maximum progress
    if agent_has_finished(Game, state, agent=0) and not agent_has_finished(Game, state, agent=1): #agent 0 is ahead
        return [1,0]
    elif not agent_has_finished(Game, state, agent=0) and agent_has_finished(Game, state, agent=1): #agent 1 is ahead
        return [0,1]
    else: #draw
        return [0,0]
    
def angle_to_goal(state, terminal_state):
    # state [x,y,theta]
    # terminal_state [x,y]
    x_now = state[0]
    y_now = state[1]
    theta_now = state[2]
    x_goal = terminal_state[0]
    y_goal = terminal_state[1]
    angle_to_goal = np.arctan2(y_goal-y_now, x_goal-x_now) - theta_now
    return angle_to_goal

def distance(state_0, state_1):
    # state = [x, y, theta]
    x1 = state_0[0]
    y1 = state_0[1]
    x2 = state_1[0]
    y2 = state_1[1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def get_agent_progress(centerline, prev_state, next_state):
    # state = [x, y, theta]
    # action = [speed, angular_speed]
    prev_closest_coordinate = find_closest_waypoint(centerline, prev_state)
    next_closest_coordinate = find_closest_waypoint(centerline, next_state)

    progress = next_closest_coordinate[-1] - prev_closest_coordinate[-1] # extract progress value
    return progress

def get_agent_advancement(centerline, state):
    closest_coordinate = find_closest_waypoint(centerline, state)
    return closest_coordinate[-1]

def find_closest_waypoint(centerline, state):
    closest_waypoint = None
    min_distance = float('inf')

    for waypoint in centerline:
        dist = distance(state, waypoint[:2])
        if dist < min_distance:
            min_distance = dist
            closest_waypoint = waypoint
    return closest_waypoint

def get_env_progress(Game, state):
    if Game.env.env_centerline is not None:
        closest_waypoint = None
        min_distance = float('inf')

        for waypoint in Game.env.env_centerline:
            dist = distance(state, waypoint[:2])
            if dist < min_distance:
                min_distance = dist
                closest_waypoint = waypoint
        return closest_waypoint[-1]
    else:
        return None

# Gaussian function
def denormalized_gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
