import os
import time
import glob
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import sys
import multiprocessing
import matplotlib.gridspec as gridspec

from common import *
from environment_utilities import *
from networkx_utilities import open_tree_from_file


class FigureV0:
    def __init__(self):
        gs_kw = dict(width_ratios=[1, 2], height_ratios=[1])
        self.fig, self.axd = plt.subplot_mosaic([['tree', 'trajectory']], gridspec_kw=gs_kw, figsize=(12, 6), layout='constrained')
        self.fig.suptitle("MCTS Tree and Trajectory")
        self.ax0 = self.axd['tree']
        self.ax1 = self.axd['trajectory']

    def clear_ax(self):
        self.ax0.clear()
        self.ax1.clear()

    def visualize_tree(self):
        tree = get_last_tree()

        pos = graphviz_layout(tree, prog="twopi")
        labels = {node: "n:{}".format(node._number_of_visits)+"\n"+"X0:{}\nX1:{}".format(round(node.X(agent=0),0), round(node.X(agent=1),0)) for node in tree.nodes}

        # Create a list of node colors
        node_colors = []
        cmap = plt.get_cmap('Oranges')
        for node in tree.nodes:
            """if any(node == parent._next_child for parent in tree.predecessors(node)):
                node_colors.append("orange")
            else:"""
            node_colors.append('lightgrey')

        #logging.debug('Node colors: %s', node_colors)
        node_colors = np.array(node_colors, dtype=object)
        nx.draw(tree, pos=pos, labels=labels, ax=self.ax0, with_labels=False, node_size=10, font_size=8, node_color=node_colors)

    def update_trajectory(self, env=None):
        # plot dynamic rollout data
        """try:
            data_rollout = pd.read_csv(path_to_rollout_curr)
            x0_rollout, y0_rollout, x1_rollout, y1_rollout = data_rollout['x0'], data_rollout['y0'], data_rollout['x1'], data_rollout['y1']
            self.ax1.plot(x0_rollout, y0_rollout, "darkturquoise", label='Rollout Agent 0')
            self.ax1.plot(x1_rollout, y1_rollout, "salmon", label='Rollout Agent 1')
        except:
            data_rollout = pd.read_csv(path_to_rollout_last)
            x0_rollout, y0_rollout, x1_rollout, y1_rollout = data_rollout['x0'], data_rollout['y0'], data_rollout['x1'], data_rollout['y1']
            self.ax1.plot(x0_rollout, y0_rollout, "darkturquoise", label='Rollout Agent 0')
            self.ax1.plot(x1_rollout, y1_rollout, "salmon", label='Rollout Agent 1')
            pass"""

        # plot longterm rollout data as points being explored with color dependend on timehorizon
        try:
            data_longterm = pd.read_csv(path_to_rollout_last)
            x0_longterm, y0_longterm, theta0_longterm, x1_longterm, y1_longterm, theta1_longterm, timehorizon = data_longterm['x0'], data_longterm['y0'], data_longterm['theta0'] , data_longterm['x1'], data_longterm['y1'], data_longterm['theta1'], data_longterm['timehorizon']

            # Store visited configurations and number of visits
            visited_configurations = {}
            for i in range(len(x0_longterm)):
                configuration0 = (0, round(x0_longterm[i], 1), round(y0_longterm[i], 1), round(theta0_longterm[i], 1), timehorizon[i])
                configuration1 = (1, round(x1_longterm[i], 1), round(y1_longterm[i], 1), round(theta1_longterm[i], 1), timehorizon[i])

                # count number of similar configurations
                if configuration0 in visited_configurations:
                    visited_configurations[configuration0] += 1
                else:
                    visited_configurations[configuration0] = 1
                if configuration1 in visited_configurations:
                    visited_configurations[configuration1] += 1
                else:
                    visited_configurations[configuration1] = 1

            max_visits = max(visited_configurations.values(), key=lambda x: x)

            # Plot visited configurations as triangles with scaled sizes
            for configuration, visits in visited_configurations.items():

                size = np.log(1+visits/max_visits) * 0.5  # Scale the size based on the number of visits

                if configuration[0] == 0 and configuration[-1] == timehorizon.max():
                    _, x0, y0, theta0, timehorizon = configuration
                    arrow_dx0 = 0.5 * np.cos(theta0)
                    arrow_dy0 = 0.5 * np.sin(theta0)
                    #self.ax1.plot(x0, y0, marker='^', markersize=size, c='darkturquoise', alpha=0.5, label='Longterm Agent 0')
                    self.ax1.arrow(x0, y0, arrow_dx0, arrow_dy0, color='darkturquoise', alpha=0.5, width=size)
                elif configuration[0] == 1 and configuration[-1] == timehorizon.max():
                    _, x1, y1, theta1, timehorizon = configuration
                    arrow_dx1 = 0.5 * np.cos(theta1)
                    arrow_dy1 = 0.5 * np.sin(theta1)
                    #self.ax1.plot(x1, y1, marker='v', markersize=size, c='salmon', alpha=0.5, label='Longterm Agent 1')
                    self.ax1.arrow(x1, y1, arrow_dx1, arrow_dy1, color='salmon', alpha=0.5, width=size)
        except:
            print("Longterm data print error")
            pass

        # plotting the trajectory of two agents on a 2D-plane and connecting states with a line and labeling states with timestep | trajectory: list of states [x1, y1, x2, y2, timestep]
        trajectory = pd.read_csv(path_to_global_state)

        x0_trajectory = trajectory['x0']
        y0_trajectory = trajectory['y0']
        theta0_trajectory = trajectory['theta0']
        x1_trajectory = trajectory['x1']
        y1_trajectory = trajectory['y1']
        theta1_trajectory = trajectory['theta1']
        timestep_trajectory = trajectory['timestep']

        self.ax1.plot(x0_trajectory, y0_trajectory, "bo-", label='Trajectory Agent 0')
        self.ax1.plot(x1_trajectory, y1_trajectory, "ro-", label='Trajectory Agent 1')
        # annotate timesteps
        for i in range(len(trajectory)):
            self.ax1.annotate(timestep_trajectory[i], (x0_trajectory[i], y0_trajectory[i]), color='b', textcoords="offset points", xytext=(0,10))
            self.ax1.annotate(timestep_trajectory[i], (x1_trajectory[i], y1_trajectory[i]), color='r', textcoords="offset points", xytext=(-10,0))
        # plot pixel environment
        self.ax1.imshow(env.get_current_grid(timestep_trajectory.iloc[-1])['grid'], cmap='binary')

        # plot orientations as arrows
        arrow_length = 0.5
        for i in range(len(trajectory)):
            arrow_dx0 = arrow_length * np.cos(theta0_trajectory[i])
            arrow_dy0 = arrow_length * np.sin(theta0_trajectory[i])
            arrow_dx1 = arrow_length * np.cos(theta1_trajectory[i])
            arrow_dy1 = arrow_length * np.sin(theta1_trajectory[i])

            self.ax1.arrow(x0_trajectory[i], y0_trajectory[i], arrow_dx0, arrow_dy0, color='b', width=0.05, zorder=10)
            self.ax1.arrow(x1_trajectory[i], y1_trajectory[i], arrow_dx1, arrow_dy1, color='r', width=0.05, zorder=10)

def get_last_tree():
        list_of_files = glob.glob(path_to_tree.format("*")) # * means all if need specific format then *.csv
        try:
            latest_file = max(list_of_files, key=os.path.getctime)
            print("Latest file: {}".format(latest_file))
            tree = open_tree_from_file(latest_file)
        except:
            print("No tree file found")
            tree = nx.DiGraph()
            pass
        return tree

def plot_together(i, figplot, env=None, stop_event=None, animation_container=None):
    figplot.clear_ax()
    figplot.visualize_tree()
    figplot.update_trajectory(env)

    figplot.ax0.set_title("MCTS Tree with {} iterations".format(MCTS_params['num_iter']))
    figplot.ax1.set_title("Trajectory")
    figplot.ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, framealpha=0.5)
    figplot.ax1.set_xlim([0, env.get_current_grid(0)['x_max']+1])
    figplot.ax1.set_ylim([0, env.get_current_grid(0)['y_max']+1][::-1]) # invert y-axis to fit to the environment defined in the numpy array

    if stop_event and stop_event.is_set():
        animation_container[0].event_source.stop() # Stop the animation
        plt.close(figplot.fig)
        sys.exit()