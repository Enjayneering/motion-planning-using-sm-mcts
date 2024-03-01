import numpy as np

# import modules
import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS')
from utilities.common_utilities import get_max_timehorizon
from utilities.environment_utilities import *

# TODO: DONE

class Environment:
    # define 2D environment with continuous free space and discretized obstacle space
    def __init__(self, config_environment, agent_indices=None):
        self.config_environment = config_environment
        self.dynamic_occup_grid = self.get_dynamic_occup_grid(self.config_environment['env_timesteps'][0])
        
        self.env_global_grid = get_global_grid(self.config_environment['env_timesteps'])
        self.finish_line = get_finish_line(self.config_environment['env_timesteps'])

        self.agents_in_env = get_agents_in_env(self.config_environment['env_agent_configs'], agent_indices) # store occupancy grids of all agents in environment
        self.joint_init_state = self.get_joint_init_state()
        self.joint_terminal_state = self.get_joint_terminal_state()

        self.env_centerline = get_env_centerline(config_environment['env_timesteps'], metric_dist=self.config_environment['stepsize_progress'])
        self.agent_progress_lines = self.get_agent_progress_lines(self.config_environment['env_agent_configs'], metric_dist=self.config_environment['stepsize_progress'])

    def get_dynamic_occup_grid(self, env_def):    
        env_list = {}

        for time_trigger, grid in env_def.items():
            env_list[time_trigger] = {
                            'grid': get_occupancy_grid(grid),
                            'x_min': self.get_x_min(grid),
                            'y_min': self.get_y_min(grid),
                            'x_max': self.get_x_max(grid),
                            'y_max': self.get_y_max(grid),}
        return env_list
        
    def get_agent_progress_lines(self, metric_dist=0.5):
        prog_lines={}

        for agent_ix, occupancy_grid in self.agents_in_env.items():
            if np.any(occupancy_grid == 8):
                centerline = []

                agent_y_init, agent_x_init = np.where(occupancy_grid == 0)
                
                centerline_y, centerline_x = np.where((occupancy_grid == 8) | (occupancy_grid == 1)) # keypoints of centerline in the grid
                centerline_y = list(centerline_y)
                centerline_x = list(centerline_x)
                centerline_coordinates = [(x, y) for x, y in zip(centerline_x, centerline_y)]
                point_y = agent_y_init[0]
                point_x = agent_x_init[0]

                while len(centerline_coordinates) > 0:
                    centerline.append([point_x, point_y])
                    point_x, point_y = get_closest_gridpoint(point=[point_x, point_y], list_of_points= centerline_coordinates)
                    centerline_coordinates.remove((point_x, point_y))
                centerline.append([point_x, point_y])
                
                prog_lines[agent_ix] = get_interpolated_centerline_KD_tree(centerline, metric_dist=metric_dist)
            else:
                print("Progress line is not defined for this environment")
                return None
        return prog_lines
    
    

        """if any([char in grid for char in ['x']]):
            occupancy_grid_define = grid.replace('.', '9').replace('#', '9').replace('S', '0').replace('G','1').replace('+', '9').replace('x', '8').replace('F', '9')
            lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
            transformed_grid = [list(map(int, line)) for line in lines]
            occupancy_grid = np.array(transformed_grid)  

            centerline = []

            centerline_y, centerline_x = np.where((occupancy_grid == 8) | (occupancy_grid == 1)) # keypoints of centerline in the grid
            centerline_y = list(centerline_y)
            centerline_x = list(centerline_x)
            centerline_coordinates = [(x, y) for x, y in zip(centerline_x, centerline_y)]
            point_x = centerline_coordinates[0][0]
            point_y = centerline_coordinates[0][-1]

            while len(centerline_coordinates) > 0:
                point_x, point_y = get_closest_gridpoint(point=[point_x, point_y], list_of_points= centerline_coordinates)
                centerline_coordinates.remove((point_x, point_y))
                centerline.append([point_x, point_y])
            
            env_centerline = get_interpolated_centerline_KD_tree(centerline, metric_dist=metric_dist)
        else:
            print("Centerline is not defined for this environment")
            return None
        return env_centerline"""
    
    def get_current_grid(self, timestep):
        for grid_timestep in reversed(self.dynamic_grid):
            if grid_timestep <= timestep:
                current_grid = self.dynamic_grid[grid_timestep]
                break
        return current_grid

    def get_x_min(self, grid):
            static_grid = self.get_occupancy_grid(grid)
            min_x = 0
            for column in range(static_grid.shape[1]):
                if any(static_grid[:,min_x] == 0):
                    #print("min progress: {}".format(min_x))
                    return min_x
                else:
                    min_x += 1

    def get_y_min(self, grid):
        static_grid = self.get_occupancy_grid(grid)
        min_y = 0

        for row in range(static_grid.shape[0]):
            if any(static_grid[min_y,:] == 0):
                #print("min progress: {}".format(min_y))
                return min_y
            else:
                min_y += 1

    def get_x_max(self, grid):
        static_grid = self.get_occupancy_grid(grid)
        max_x = static_grid.shape[1]-1

        for column in range(max_x):
            if any(static_grid[:,max_x] == 0):
                #print("max progress: {}".format(max_x))
                return max_x
            else:
                max_x -= 1

    def get_y_max(self, grid):
        static_grid = self.get_occupancy_grid(grid)
        max_y = static_grid.shape[0]-1

        for row in range(max_y):
            if any(static_grid[max_y,:] == 0):
                #print("max progress: {}".format(max_y))
                return max_y
            else:
                max_y -= 1
    
    def get_joint_init_state(self):
        joint_init_state = np.zeros((len(self.agents_in_env), 3))

        for ix, occupancy_grid in self.agents_in_env.items():

            agent_y_init, agent_x_init = np.where(occupancy_grid == 0)  # single value

            agent_y_next, agent_x_next = np.where(occupancy_grid == 8)  # multiple values
            agent_y_next = list(agent_y_next)
            agent_x_next = list(agent_x_next)
            agent_next_list = [(x, y) for x, y in zip(agent_x_next, agent_y_next)]

            closest_x_next, closest_y_next = get_closest_gridpoint(point=[agent_x_init[0], agent_y_init[0]], list_of_points= agent_next_list)
            theta_init = np.arctan2(closest_y_next-agent_y_init[0], closest_x_next-agent_x_init[0])

            joint_init_state[ix] = [agent_x_init[0], agent_y_init[0], theta_init]

        return joint_init_state
    
    def get_joint_terminal_state(self):
        joint_goal_state = np.zeros((len(self.agents_in_env), 3))

        for ix, occupancy_grid in self.agents_in_env.items():

            if any((occupancy_grid == 1).flatten()):
                agent_y_goal, agent_x_goal = np.where(occupancy_grid == 1)  # single value

                agent_y_prev, agent_x_prev = np.where(occupancy_grid == 8)  # multiple values
                agent_y_prev = list(agent_y_prev)
                agent_x_prev = list(agent_x_prev)
                agent_previous_list = [(x, y) for x, y in zip(agent_x_prev, agent_y_prev)]

                closest_x_prev, closest_y_prev = get_closest_gridpoint(point=[agent_x_goal[0], agent_y_goal[0]], list_of_points=agent_previous_list)
                theta_goal = np.arctan2(agent_y_goal[0]-closest_y_prev, agent_x_goal[0]-closest_x_prev)

                joint_goal_state[ix] = [agent_x_goal[0], agent_y_goal[0], theta_goal]
            else:
                joint_goal_state[ix] = [None, None, None]

        return joint_goal_state
    
    def get_game_timehorizon(self, config_global, curr_state=None):
        """
        joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
        """
        if curr_state is None:
            curr_state = self.joint_init_state
        terminal_state = self.joint_terminal_state

        min_time = self.get_min_time_to_complete(curr_state, terminal_state, config_global)

        max_game_timehorizon = int(config_global['alpha_terminal'] * min_time)+1
        #print("Max game timehorizon: {}".format(max_game_timehorizon))
        return max_game_timehorizon

    def get_min_time_to_complete(self, curr_state, terminal_state, config_global):
        """
        joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
        """
        # Extract the positions of all agents from the state arrays
        curr_positions = curr_state[:, :2]
        terminal_positions = terminal_state[:, :2]

        # Compute the Euclidean distances from the current to the terminal positions
        distances = np.linalg.norm(curr_positions - terminal_positions, axis=1)

        # Build an array of the maximum velocities of all agents
        max_velocities = np.array([np.max(config['assumptions'][agent_id]['vel']) for agent_id, config in config_global['config_agents'].items()])

        # Compute the times each agent needs to reach the terminal state
        times = distances / max_velocities

        return np.max(times)



def get_global_grid(env_global_def):
        global_grid = env_global_def.replace('.', '0').replace('#', '1').replace('+', '1').replace('x', '8').replace('F', '7')
        lines = [line.replace(' ', '') for line in global_grid.split('\n') if line]
        transformed_grid = [list(map(int, line)) for line in lines]
        global_grid = np.array(transformed_grid)
        return global_grid

def get_occupancy_grid(grid):
    # return array of zeros and ones
    occupancy_grid_define = grid.replace('.', '0').replace('#', '1').replace('+', '2').replace('F', '0').replace('x', '0')
    lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
    transformed_grid = [list(map(int, line)) for line in lines]
    occupancy_grid = np.array(transformed_grid)
    return occupancy_grid

def get_agents_in_env(env_agent_configs, agent_indices=None):
    agent_configs = {}
    for agent_ix, grid in env_agent_configs.items():
        occupancy_grid = grid.replace('.', '9').replace('#', '9').replace('+', '9').replace('S','0').replace('G', '1').replace('x', '8')
        lines = [line.replace(' ', '') for line in occupancy_grid.split('\n') if line]
        transformed_grid = [list(map(int, line)) for line in lines]
        occupancy_grid = np.array(transformed_grid)     

        # change perspective
        if agent_indices is not None:
            agent_configs[agent_indices[agent_ix]] = occupancy_grid
        else:
            agent_configs[agent_ix] = occupancy_grid
    return agent_configs

def get_env_centerline(env_global_grid, metric_dist=0.5):
    """
    returns a centerline as dict of {KDtree: tree, progress: progress}
    """
    env_centerline_points = []
    centerline_y, centerline_x = np.where((env_global_grid == 8)) # keypoints of centerline in the grid
    centerline_y = list(centerline_y)
    centerline_x = list(centerline_x)
    centerline_coordinates = [(x, y) for x, y in zip(centerline_x, centerline_y)]
    point_x = centerline_coordinates[0][0]
    point_y = centerline_coordinates[0][-1]

    while len(centerline_coordinates) > 0:
        point_x, point_y = get_closest_gridpoint(point=[point_x, point_y], list_of_points= centerline_coordinates)
        centerline_coordinates.remove((point_x, point_y))
        env_centerline_points.append([point_x, point_y])
    
    env_centerline = get_interpolated_centerline_KD_tree(env_centerline_points, metric_dist=metric_dist)

    return env_centerline

def get_finish_line(global_grid):
    if any((global_grid == 7).flatten()):
        print("Running in Racing mode with finish line!")
        finish_line = np.where(global_grid == 7)[1] #column value
        return finish_line[0]
    else:
        print("No finish line defined")
        return None