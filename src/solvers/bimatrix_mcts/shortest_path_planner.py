import matplotlib.pyplot as plt
import numpy as np
import os

from .a_star import *

class ShortestPathPlanner:
    def __init__(self, env, action_set, curr_state=None, opponent_state=None, start_timestep=0, dt=1, ix_agent=0):
        self.agent = ix_agent
        self.start_timestep = start_timestep
        #self.dt = model_conf.confdict['delta_t']
        self.dt = dt
        #self.env = Environment(street.confdict[0])
        self.env = env
        #self.action_set = {'velocity': [round(val, 2) for val in agent_conf.confdict[ix_agent]['velocity'] if val != 0], 
        #                   'ang_velocity': [round(val, 2) for val in agent_conf.confdict[ix_agent]['ang_velocity']]}
        self.action_set = {'velocity': [round(val,2) for val in action_set['velocity'] if val >= 0],
                           'ang_velocity': [round(val,2) for val in action_set['ang_velocity']]}

        if ix_agent == 0:
            if curr_state is not None:
                self.start = curr_state
            else:
                self.start = [self.env.init_state['x0'], self.env.init_state['y0'], self.env.init_state['theta0']]
                print("Start state: {}".format(self.start))
            self.goal = [self.env.goal_state['x0'], self.env.goal_state['y0'], self.env.goal_state['theta0']]
        elif ix_agent == 1:
            if curr_state is not None:
                self.start = curr_state
            else:
                self.start = [self.env.init_state['x1'], self.env.init_state['y1'], self.env.init_state['theta1']]
            self.goal = [self.env.goal_state['x1'], self.env.goal_state['y1'], self.env.goal_state['theta1']]

        self.opp_state = opponent_state
    
    def get_trajectories(self, num_trajectories, scale_dense=0.0, block_from_ends=0.0):
        n_traj = 0
        min_length = 0
        #min_timesteps = 0
        constraints = []

        
        trajectories = {'states': [], 'actions': [], 'times': []}

        while n_traj < num_trajectories:
            # compute multiple paths
            h_values = compute_heuristics(self.env.dynamic_grid[0]['grid'], tuple(self.goal[:2]))
            #h_values = compute_heuristics_goal_line(self.env.dynamic_grid[0]['grid'], self.goal[0], self.goal[1])

            # add constraints regarding position of the other agent
            #opp_position_now = tuple([int(val) for val in self.opp_state[:2]])
            #constraints = [{'timestep': 0, 'loc': opp_position_now, 'goal_dist': h_values[opp_position_now]}]
            
            #try:
            #    opp_position_next = tuple([int(val) for val in mm_unicycle(self.opp_state, [1.0,0.0], self.dt)[:2]])
            #    constraints.append({'timestep': 1, 'loc': opp_position_next, 'goal_dist': h_values[opp_position_next]})
            #except:
            #    print("next state collides")

            action_path, state_path, time_path = a_star(self.env, tuple(self.start), tuple(self.goal), h_values, constraints=constraints, action_set=self.action_set, delta_t=self.dt, start_timestep=0, scale_dense=scale_dense)
            
            if action_path is not None:
                trajectories['states'].append(state_path)
                trajectories['actions'].append(action_path)
                trajectories['times'].append(time_path)
                n_traj += 1
            
                # add random constraints to get variety of paths
                min_length = max(min_length, len(state_path))
                #min_timesteps = max(min_timesteps, time_path[-1])
                index = int(min_length*block_from_ends)

            grid = self.env.dynamic_grid[0]['grid']
            
            #constraints = add_obstacle_constraints(grid, h_values, n=10) # n: number of obstacles
            
            # add trajectory constrints?
            #print(constraints)

            """for timestep, state in zip(time_path[index:-index], state_path[index:-index]):
                constraints.append({'timestep': timestep, 'loc': state, 'goal_dist': h_values[state[:2]]})
                #print("Constraint: {}".format({'timestep': timestep, 'loc': state, 'goal_dist': h_values[state[:2]]}))"""

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        plt.figure()
        plt.gca().invert_yaxis()

        for i, trajectory in enumerate(trajectories['states']):
            xs = [state[0] for state in trajectory]
            ys = [state[1] for state in trajectory]
            thetas = [state[2] for state in trajectory]
            
            plt.plot(xs, ys, color=colors[i % len(colors)])
            
            # Add arrows to indicate direction
            for j in range(len(xs)):
                dx = np.cos(thetas[j])
                dy = np.sin(thetas[j])
                plt.arrow(xs[j], ys[j], dx, dy, color=colors[i % len(colors)], head_width=0.05, head_length=0.1)
        plt.savefig(os.path.join(path_to_test_data, f"{self.start_timestep}_trajectories_{self.agent}.png"))
        plt.close()
        
        return trajectories
    
def add_constraints(constraints, state_traj, time_traj, h_values, agent):
    for state, timestep in zip(state_traj, time_traj):
        constraints.append({'timestep': timestep, 'loc': state, 'goal_dist': h_values[state[:2]], 'agent': agent})
    return constraints

def add_obstacle_constraints(grid, h_values, n):
    """
    Adds n obstacles to the environment.

    Parameters:
        env: 2D numpy array representing the environment.
        n: Number of obstacles to add.
    """
    height, width = grid.shape
    temp_grid = np.zeros(grid.shape)

    constraints = []
    for _ in range(n):
        # Generate random coordinates for the obstacle
        x, y = np.random.randint(0, width), np.random.randint(0, height)

        # Check if there is already an obstacle at this location
        if temp_grid[y, x] == 1:
            continue

        # Check if the new obstacle is adjacent to any existing obstacles
        if x > 0 and temp_grid[y, x - 1] == 1:
            continue
        if x < width - 1 and temp_grid[y, x + 1] == 1:
            continue
        if y > 0 and temp_grid[y - 1, x] == 1:
            continue
        if y < height - 1 and temp_grid[y + 1, x] == 1:
            continue

        # If the location is valid, add the point to the constraints
        temp_grid[y, x] = 1
        #for t in range(min_timesteps):

        # neglect timesteps, because we treat obstacles static
        constraints.append({'timestep': 1, 'loc': (x,y), 'goal_dist': h_values[(x,y)]})
    print("Sampled Obstacles: \n{}".format(temp_grid))
    return constraints

if __name__ == "__main__":
    SPPlanner = ShortestPathPlanner(ix_agent=0)
    trajectories = SPPlanner.get_trajectories(num_trajectories=30, scale_dense=0)
    print(trajectories['states'])
    list_of_state_trajectories = trajectories['states']

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.figure()

    for i, trajectory in enumerate(list_of_state_trajectories):
        xs = [state[0] for state in trajectory]
        ys = [state[1] for state in trajectory]
        thetas = [state[2] for state in trajectory]
        
        plt.plot(xs, ys, color=colors[i % len(colors)])
        
        # Add arrows to indicate direction
        for j in range(len(xs)):
            dx = np.cos(thetas[j])
            dy = np.sin(thetas[j])
            plt.arrow(xs[j], ys[j], dx, dy, color=colors[i % len(colors)], head_width=0.05, head_length=0.1)

    plt.show()


    
