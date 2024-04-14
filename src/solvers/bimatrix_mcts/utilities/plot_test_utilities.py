import os
import sys

import glob
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
import pandas as pd
import sys

from .common_utilities import *
from .environment_utilities import *
from .networkx_utilities import open_tree_from_file


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
        #labels = {node: "n:{}".format(node._number_of_visits)+"\n"+"X0:{}\nX1:{}".format(round(node.X(agent=0),0), round(node.X(agent=1),0)) for node in tree.nodes}

        # Create a list of node colors based on the number of visits
        node_colors = []
        cmap = plt.get_cmap('coolwarm')

        def count_predecessors(node):
            count = 0
            while True:
                predecessors = list(tree.predecessors(node))
                if len(predecessors) == 0:
                    break
                count += 1
                node = predecessors[0]
            return count
        
        # paint nodes colorful on every depth level
        try:
            depth = 0
            for node in tree.nodes:
                if count_predecessors(node) > depth:
                    depth = count_predecessors(node)

            max_visits = []
            for d in range(depth+1):
                visit_counts = [node.n() for node in tree.nodes if count_predecessors(node) == d]
                max_visits.append(np.max(visit_counts))

            global_max_visits = max_visits[1] #np.max(max_visits)

            for node in tree.nodes:
                depth = count_predecessors(node)
                if depth != 0:  # for nodes on that level
                    visit_count = node.n()
                    #color = cmap(visit_count / max_visits[depth]) # depth 1 or depth
                    norm = colors.LogNorm(vmin=1, vmax=global_max_visits)
                    normalized_visit_count = norm(visit_count)
                    color = cmap(normalized_visit_count)
                    node_colors.append(color)
                else:
                    node_colors.append('yellow')

            #logging.debug('Node colors: %s', node_colors)
            node_colors = np.array(node_colors, dtype=object)

            # Plot the node color legend
            if not hasattr(self, 'colorbar'):
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.LogNorm(vmin=1, vmax=global_max_visits))
                sm.set_array([])
                self.colorbar = self.fig.colorbar(sm, ax=self.ax0, label='Number of Visits')
            nx.draw(tree, pos=pos, labels=None, ax=self.ax0, with_labels=False, node_size=12, font_size=8, node_color=node_colors)
        except:
            pass

        
        

    def update_trajectory(self, Game):
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
                    self.ax1.arrow(x0, y0, arrow_dx0, arrow_dy0, color='salmon', alpha=0.5, width=size)
                elif configuration[0] == 1 and configuration[-1] == timehorizon.max():
                    _, x1, y1, theta1, timehorizon = configuration
                    arrow_dx1 = 0.5 * np.cos(theta1)
                    arrow_dy1 = 0.5 * np.sin(theta1)
                    #self.ax1.plot(x1, y1, marker='v', markersize=size, c='salmon', alpha=0.5, label='Longterm Agent 1')
                    self.ax1.arrow(x1, y1, arrow_dx1, arrow_dy1, color='darkturquoise', alpha=0.5, width=size)
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

        # plot goal configuration
        #plot_goal_states(self.ax1, Game.env.goal_state)

        # plot centerlines
        plot_centerline(self.ax1, Game.env.centerlines, agent=0, color='r', alpha=0.5)
        plot_centerline(self.ax1, Game.env.centerlines, agent=1, color='b', alpha=0.5)

        self.ax1.plot(x0_trajectory, y0_trajectory, "ro-", label='Trajectory Agent 0')
        self.ax1.plot(x1_trajectory, y1_trajectory, "bo-", label='Trajectory Agent 1')
        # annotate timesteps
        for i in range(len(trajectory)):
            self.ax1.annotate(timestep_trajectory[i], (x0_trajectory[i], y0_trajectory[i]), color='r', textcoords="offset points", xytext=(0,10))
            self.ax1.annotate(timestep_trajectory[i], (x1_trajectory[i], y1_trajectory[i]), color='b', textcoords="offset points", xytext=(-10,0))
        # plot pixel environment
        ix = timestep_trajectory.iloc[-1]
        self.ax1.imshow(Game.env.get_current_grid(timestep_trajectory.iloc[-1])['grid'], cmap='binary')

        # plot orientations as arrows
        arrow_length = 0.5
        for i in range(len(trajectory)):
            arrow_dx0 = arrow_length * np.cos(theta0_trajectory[i])
            arrow_dy0 = arrow_length * np.sin(theta0_trajectory[i])
            arrow_dx1 = arrow_length * np.cos(theta1_trajectory[i])
            arrow_dy1 = arrow_length * np.sin(theta1_trajectory[i])

            self.ax1.arrow(x0_trajectory[i], y0_trajectory[i], arrow_dx0, arrow_dy0, color='r', width=0.05, zorder=10)
            self.ax1.arrow(x1_trajectory[i], y1_trajectory[i], arrow_dx1, arrow_dy1, color='b', width=0.05, zorder=10)

def get_last_tree():
        list_of_files = glob.glob(path_to_trees+"/tree_{}.csv".format("*")) # * means all if need specific format then *.csv
        try:
            latest_file = max(list_of_files, key=os.path.getctime)
            print("Latest file: {}".format(latest_file))
            tree = open_tree_from_file(latest_file)
        except:
            print("No tree file found")
            tree = nx.DiGraph()
            pass
        return tree

def plot_centerline(ax, centerlines, agent=0, color='black', alpha=1):
    x_centerline = [c[0] for c in centerlines[agent]]
    y_centerline = [c[1] for c in centerlines[agent]]
    ax.plot(x_centerline, y_centerline, "x--", color=color, alpha=alpha, label=f'Centerline Agent {agent}', zorder=1)

def plot_goal_states(ax, goal_state):
    if goal_state is not None:
        x_goal_0 = goal_state['x0']
        y_goal_0 = goal_state['y0']
        x_goal_1 = goal_state['x1']
        y_goal_1 = goal_state['y1']
        ax.plot(x_goal_0, y_goal_0, 'o', color='red', markersize=5, label='0')
        ax.plot(x_goal_1, y_goal_1, 'o', color='blue', markersize=5, label='1')
    else:
        pass

def plot_together(i, figplot, Game, stop_event, animation_container):
    print("Plotting iteration {}".format(i))
    figplot.clear_ax()

    # plot tree and trajectory
    figplot.visualize_tree()
    figplot.update_trajectory(Game)

    figplot.ax0.set_title("MCTS Tree with {} iterations".format(Game.MCTS_params['num_iter']))
    figplot.ax1.set_title("Trajectory")
    figplot.ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.15), ncol=2, fancybox=True, framealpha=0.5)
    xmax = Game.env.x_max
    ymax = Game.env.y_max
    #print("xmax: {}, ymax: {}".format(xmax, ymax))
    figplot.ax1.set_xlim([-0.5, xmax+0.5])
    figplot.ax1.set_ylim([-0.5, ymax+0.5][::-1]) 

    if stop_event and stop_event.is_set():
        if animation_container[0] is not None:
            animation_container[0].event_source.stop() # Stop the animation
        plt.close(figplot.fig)
        sys.exit()

def plot_independent(Game, stop_event):
    figplot = FigureV0()
    interval = 1000

    frames_max = get_max_timehorizon(Game)*Game.config.num_iter*interval*10

    animation_container = [None] # Container to store the animation object
    animation = FuncAnimation(figplot.fig, plot_together, fargs=(figplot, Game, stop_event, animation_container), frames=frames_max, interval=interval)
    if Game.config.feature_flags["run_mode"]['live-plot']:
        plt.show()
    else:
        animation_container[0] = animation  # Store the animation in the container
        animation_container[0].save(os.path.join(path_to_results, Game.name + ".mp4"), writer='ffmpeg', fps=5)
        print("Finished plotting")
