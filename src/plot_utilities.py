import os
import glob
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import sys
from matplotlib.ticker import AutoMinorLocator

from common import *
from environment_utilities import *
from matplotlib.patches import Rectangle
from kinodynamic_utilities import distance, mm_unicycle

def plot_single_run(Game, result_dict, path_to_experiment, timestep=None, main_agent=0):
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
    plotCOS(ax, x_orig=0, y_orig=0, scale=1, colormap=colormap)
                 
    # plot finish line
    plot_finish_line(ax, finish_line=finish_line, ymax=ymax)
    #plot_goal_states(ax, goal_state=Game.env.goal_state, agent=0, color=colormap['red'])
    #plot_goal_states(ax, goal_state=Game.env.goal_state, agent=1, color=colormap['blue'])
    plot_centerline(ax, centerlines=Game.env.centerlines, agent=0, color=colormap['red'], alpha=0.5)
    plot_centerline(ax, centerlines=Game.env.centerlines, agent=1, color=colormap['blue'], alpha=0.5)
    
    # plot trajectories
    if main_agent == 0:
        plot_trajectory(ax, trajectory_0, facecolor=colormap['red'], edgecolor=colormap['darkred'] ,label='Agent 0 (Us)', zorder=100, alpha=1)
        plot_trajectory(ax, trajectory_1, facecolor=colormap['blue'], edgecolor=colormap['darkblue'] , label='Agent 1 (Opponent)', zorder=50, alpha=1)
    elif main_agent == 1:
        plot_trajectory(ax, trajectory_0, facecolor=colormap['red'], edgecolor=colormap['darkred'] ,label='Agent 0 (Us)', zorder=50, alpha=1)
        plot_trajectory(ax, trajectory_1, facecolor=colormap['blue'], edgecolor=colormap['darkblue'] , label='Agent 1 (Opponent)', zorder=100, alpha=1) 
    
    
    # plot legend
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, fancybox=True, framealpha=0.5, bbox_transform=fig.transFigure) #title='Trajectory of'
    ax.set_aspect('equal', 'box')
    # remove x and y labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.axis('off')
    fig.tight_layout()

    # save figure
    fig.savefig(os.path.join(path_to_experiment, "trajectory{}_{}.png".format(main_agent, timestep)), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_finish_line(ax, finish_line=None, ymax=None):
    if finish_line is not None:
        # Replace dotted line with periodic square patches
        patch_width = 0.5
        patch_height = ymax

        i = 0
        while i * patch_width < ymax:
            if i % 2 == 0:
                rect = Rectangle((finish_line - patch_width, i * patch_width), patch_width, patch_width, facecolor='white', edgecolor='white')
            else:
                rect = Rectangle((finish_line, i * patch_width), patch_width, patch_width, facecolor='white', edgecolor='white')
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

def plot_trajectory(ax, trajectory, linewidth=4, facecolor=None, edgecolor=None, label=None, fontsize=4, zorder=None, alpha=None):
    xy_visited = []
    x_values = [sublist[0] for sublist in trajectory]
    y_values = [sublist[1] for sublist in trajectory]
    theta_values = [sublist[2] for sublist in trajectory]
    timesteps = [sublist[3] for sublist in trajectory]
    # plot connecting lines
    ax.plot(x_values, y_values, color=facecolor, linestyle='-', linewidth=linewidth, label=label, zorder=zorder, alpha=alpha*0.75)

    # annotate timesteps
    for i in range(len(trajectory)):
        annotate_state(ax, x_values[i], y_values[i], theta_values[i], timesteps[i], visit_count=xy_visited.count([x_values[i], y_values[i]]), facecolor=facecolor, edgecolor=edgecolor, fontsize=fontsize, zorder=zorder+1, alpha=alpha)
        xy_visited.append([x_values[i], y_values[i]])

def annotate_state(ax, x, y, theta, timestep, visit_count=0, facecolor=None, edgecolor=None, fontsize=None, zorder=None, alpha=None):
    # plot orientations as arrows
    arrow_length = 0.1+0.2*(visit_count)
    arrow_width = 0.2
    head_width = 0.4
    head_length = 0.4
    linewidth = 0.4
    annot_alignment = 0.25

    arrow_dx = arrow_length * np.cos(theta)
    arrow_dy = arrow_length * np.sin(theta)
    ax.arrow(x, y, arrow_dx, arrow_dy, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, width=arrow_width, head_width=head_width , head_length=head_length, zorder=zorder+visit_count, alpha=alpha)
    
    # annotate timestep on arrow
    ax.annotate(timestep, (x+arrow_dx+annot_alignment*head_length*np.cos(theta), y+arrow_dy+annot_alignment*head_length*np.sin(theta)), color=edgecolor, fontsize=fontsize, textcoords="offset points", xytext=(0, 0), ha='center', va='center', zorder=zorder+visit_count, alpha=alpha)

    # add cicrle in the middle of the state
    loc_circle = plt.Circle((x, y), radius=0.1, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.4, zorder=zorder+visit_count, alpha=alpha)
    ax.add_patch(loc_circle)

def plotCOS(ax, x_orig, y_orig, scale, colormap, fontsize = 8, alpha=0.5):
    thickness = 0.2
    ax.arrow(x_orig, y_orig, scale, 0, head_width=thickness, head_length=thickness, fc=colormap['grey'], ec=colormap['grey'], alpha=alpha, zorder=4)
    ax.arrow(x_orig, y_orig, 0, scale, head_width=thickness, head_length=thickness, fc=colormap['grey'], ec=colormap['grey'], alpha=alpha, zorder=4)
    ax.plot(x_orig, y_orig, 'o', color=colormap['grey'], markersize=5, alpha=alpha, zorder=5)

    ax.text(x_orig, y_orig-0.3, "(0,0)", fontsize=fontsize, ha='center', va='center', alpha=alpha, zorder=4)
    ax.text(x_orig+scale, y_orig-0.3, "x", fontsize=fontsize, ha='center', va='center', alpha=alpha, zorder=4)
    ax.text(x_orig-0.3, y_orig+scale, "y", fontsize=fontsize, ha='center', va='center', alpha=alpha, zorder=4)

def plot_map(ax, array, facecolor=None, edgecolor=None):
    # plot the racing gridmap
    for row in range(len(array)):
        for col in range(len(array[0])):
            if array[row][col] == 1: # static obstacles
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, facecolor=facecolor, edgecolor=edgecolor, alpha=1, zorder=3))
            if array[row][col] == 2: # dynamic obstacles
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, facecolor=facecolor, edgecolor=edgecolor, alpha=1, hatch="//", zorder=3))
            elif array[row][col] == 0: # free space, road
                ax.plot(col, row, 'o', color='dimgrey', markersize=1, alpha=1)
                ax.add_patch(Rectangle((col-0.5, row-0.5), 1, 1, color='grey', linewidth=0, alpha=1, zorder=1))


def get_current_grid(grid_dict, timestep):
    for grid_timeindex in reversed(grid_dict.keys()):
        if grid_timeindex <= timestep:
            current_grid = grid_dict[grid_timeindex]
            break
    occupancy_grid_define = current_grid.replace('.', '0').replace('0', '0').replace('1', '0').replace('#', '1').replace('+', '2')
    lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
    transformed_grid = [list(map(int, line)) for line in lines]
    occupancy_grid = np.array(transformed_grid)
    return occupancy_grid