import os
import glob
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import pandas as pd
from common_utilities import map, path_to_global_state, path_to_rollout_curr, path_to_data
from networkx_utilities import open_tree_from_file
import logging

#logging.basicConfig(level=logging.DEBUG)

class TreePlot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def clear_ax(self):
        self.ax.clear()
    
    def visualize_tree(self, timehorizon=1):
        tree = get_last_tree(path_to_data)

        pos = graphviz_layout(tree, prog="twopi")
        labels = {node: "Ix:{}".format(node.index)+"\n"+"n:{}".format(node._number_of_visits)+"\n" for node in tree.nodes}

        # Create a list of node colors
        node_colors = []
        cmap = plt.get_cmap('Oranges')
        for node in tree.nodes:
            if any(node == parent._next_child for parent in tree.predecessors(node)):
                color_intensity = node.state.timestep / timehorizon  # normalize to range [0, 1]
                node_colors.append(cmap(color_intensity))
            else:
                node_colors.append('lightgrey')

        #logging.debug('Node colors: %s', node_colors)
        node_colors = np.array(node_colors, dtype=object)
        nx.draw(tree, pos=pos, labels=labels, ax=self.ax, with_labels=True, node_size=1000, font_size=8, node_color=node_colors)

    def plot(self, i):
        self.clear_ax()
        self.visualize_tree()

class TrajectoryPlot:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def clear_ax(self):
        self.ax.clear()

    def plot_trajectory_dynamic_rollout(self):
        # plotting the trajectory of two agents on a 2D-plane and connecting states with a line and labeling states with timestep
        # trajectory: list of states [x1, y1, x2, y2, timestep]
        trajectory = pd.read_csv(path_to_global_state)

        x0_trajectory, y0_trajectory, x1_trajectory, y1_trajectory, timestep_trajectory = trajectory['x_0'], trajectory['y_0'], trajectory['x_1'], trajectory['y_1'], trajectory['timestep']
        self.ax.plot(x0_trajectory, y0_trajectory, "bo-", label='Trajectory Agent 0')
        self.ax.plot(x1_trajectory, y1_trajectory, "ro-", label='Trajectory Agent 1')
        # annotate timesteps
        for i in range(len(x0_trajectory)):
            self.ax.annotate(timestep_trajectory[i], (x0_trajectory[i], y0_trajectory[i]), textcoords="offset points", xytext=(0,10))
            self.ax.annotate(timestep_trajectory[i], (x1_trajectory[i], y1_trajectory[i]), textcoords="offset points", xytext=(-10,0))
        self.ax.imshow(map, cmap='binary')

        data_rollout = pd.read_csv(path_to_rollout_curr)
        x0_rollout, y0_rollout, x1_rollout, y1_rollout = data_rollout['x_0'], data_rollout['y_0'], data_rollout['x_1'], data_rollout['y_1']
        self.ax.plot(x0_rollout, y0_rollout, "darkturquoise", label='Rollout Agent 0')
        self.ax.plot(x1_rollout, y1_rollout, "salmon", label='Rollout Agent 1')

    def plot(self, i):
        self.clear_ax()
        self.plot_trajectory_dynamic_rollout()

def get_last_tree(path_to_data):
    list_of_files = glob.glob(path_to_data+"tree_*") # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    try: 
        tree = open_tree_from_file(latest_file)
    except:
        pass
    return tree

if __name__ == "__main__":
    #tree_plot = TreePlot()
    trajectory_plot = TrajectoryPlot()

    #animation_tree = FuncAnimation(tree_plot.fig, tree_plot.plot, interval=100)
    animation_trajectory = FuncAnimation(trajectory_plot.fig, trajectory_plot.plot, interval=100)

    plt.show()