import numpy as np
from scipy import interpolate

class Environment:
    # define 2D environment with continuous free space and discretized obstacle space
    def __init__(self, config=None):
        self.dynamic_grid = self.init_dynamic_grid(config['env_def'])
        self.init_state = get_init_state(config['env_raceconfig'])
        self.goal_state = get_goal_state(config['env_raceconfig'])
        self.finish_line = self.get_finish_line(config['env_def'])
        self.centerlines = self.get_centerlines(config['env_raceconfig'], metric_dist=0.5)
        self.env_centerline = self.get_env_centerline(config['env_def'], metric_dist=0.5)
        self.x_max = self.get_x_max(config['env_def'][0])
        self.y_max = self.get_y_max(config['env_def'][0])
        self.x_min = self.get_x_min(config['env_def'][0])
        self.y_min = self.get_y_min(config['env_def'][0])
    
    def _interpolate_line(self, point_list, metric_dist=0.5, k=1):
        # Combine the x and y coordinates into an array of 2D points
        points = np.array(point_list)

        # Calculate the cumulative distance along the path
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)

        # Create a spline representation of the path
        tck, u = interpolate.splprep(points.T, u=distance, s=0, k=k)

        # Interpolate the points along the path at metric_dist intervals
        #new_distance = np.arange(0, distance.max(), metric_dist)
        # including last point
        num_points = int(distance.max() / metric_dist) + 1
        new_distance = np.linspace(0, distance.max(), num_points)
        
        new_points = interpolate.splev(new_distance, tck)

        # Convert the interpolated points to a list of tuples
        interpolated_points = list(zip(*new_points))
        interpolated_points = [list(point) + [dist] for point, dist in zip(interpolated_points, new_distance)]
        return interpolated_points

    def init_dynamic_grid(self, env_def):    
        env_list = []

        for time_trigger, grid in env_def.items():
            env_list.append({'timestep': time_trigger,
                            'grid': self.get_occupancy_grid(grid),
                            'x_min': self.get_x_min(grid),
                            'y_min': self.get_y_min(grid),
                            'x_max': self.get_x_max(grid),
                            'y_max': self.get_y_max(grid),})
        return env_list
    
    def get_finish_line(self, env_def):
        grid = env_def[0]
        if any([char in grid for char in ['F']]):
            print("Running in Racing mode with finish line!")
            occupancy_grid_define = grid.replace('.', '9').replace('#', '9').replace('F', '1').replace('x', '9')
            lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
            transformed_grid = [list(map(int, line)) for line in lines]
            occupancy_grid = np.array(transformed_grid)     

            finish_line = np.where(occupancy_grid == 1)[1] #column value
            return finish_line[0]
        else:
            return None
        
    def get_centerlines(self, env_centerline, metric_dist=0.5):
        centerlines={}
        for agent in range(0,2):
            grid = env_centerline[f'{agent}']
            if any([char in grid for char in ['x']]):
                occupancy_grid_define = grid.replace('.', '9').replace('#', '9').replace('S', '0').replace('G','1').replace('+', '9').replace('x', '8')
                lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
                transformed_grid = [list(map(int, line)) for line in lines]
                occupancy_grid = np.array(transformed_grid)  

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
                
                centerlines[agent] = self._interpolate_line(centerline, metric_dist=metric_dist)
            else:
                print("Centerline is not defined for this environment")
                return None
        #print(centerlines)
        return centerlines
    
    def get_env_centerline(self, env_def, metric_dist=0.5):
        env_centerline = None
        grid = env_def[0]
        if any([char in grid for char in ['x']]):
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
            
            env_centerline = self._interpolate_line(centerline, metric_dist=metric_dist)
        else:
            print("Centerline is not defined for this environment")
            return None
    #print(centerlines)
        return env_centerline

    def get_occupancy_grid(self, grid):
        occupancy_grid_define = grid.replace('.', '0').replace('0', '0').replace('1', '0').replace('#', '1').replace('+', '2').replace('O', '0').replace('I', '0').replace('F', '0').replace('x', '0')
        lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
        transformed_grid = [list(map(int, line)) for line in lines]
        occupancy_grid = np.array(transformed_grid)
        return occupancy_grid
    
    def get_current_grid(self, timestep):
        for grid_element in reversed(self.dynamic_grid):
            if grid_element['timestep'] <= timestep:
                current_grid = grid_element
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

def get_closest_gridpoint(point=None, list_of_points=None):
    # Combine the x and y coordinates into arrays of 2D points
    init_pos = np.array([point[0], point[1]])
    next_pos = np.array(list_of_points)

    # Calculate the Euclidean distance from the initial position to each next position
    distances = np.linalg.norm(next_pos - init_pos, axis=1)

    # Find the index of the next position with the smallest distance
    closest_index = np.argmin(distances)

    # Get the closest next position
    closest_x_next, closest_y_next = next_pos[closest_index]
    return closest_x_next, closest_y_next

def get_goal_state(env_raceconfig):
    goal_state = {}
    for agent in range(0,2):
        grid = env_raceconfig[f'{agent}']
        if any([char in grid for char in ['G']]):
            print("Running in free mode with goal regions!")
            occupancy_grid_define = grid.replace('.', '9').replace('#', '9').replace('S', '0').replace('G','1').replace('+', '9').replace('x', '8')
            lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
            transformed_grid = [list(map(int, line)) for line in lines]
            occupancy_grid = np.array(transformed_grid)     

            agent_y_goal, agent_x_goal = np.where(occupancy_grid == 1) # single value

            agent_y_prev, agent_x_prev = np.where(occupancy_grid == 8) # multiple values
            agent_y_prev = list(agent_y_prev)
            agent_x_prev = list(agent_x_prev)
            agent_previous_list = [(x, y) for x, y in zip(agent_x_prev, agent_y_prev)]

            closest_x_prev, closest_y_prev = get_closest_gridpoint(point=[agent_x_goal[0], agent_y_goal[0]], list_of_points= agent_previous_list)
            theta_goal = np.arctan2(agent_y_goal[0]-closest_y_prev, agent_x_goal[0]-closest_x_prev)
            
            goal_state[f'x{agent}'] = agent_x_goal[0]
            goal_state[f'y{agent}'] = agent_y_goal[0]
            goal_state[f'theta{agent}'] = theta_goal
        else:
            return None
    return goal_state

def get_init_state(env_raceconfig):
    init_state = []
    for agent in range(0,2):
        grid = env_raceconfig[f'{agent}']
        occupancy_grid_define = grid.replace('.', '9').replace('#', '9').replace('S','0').replace('G', '1').replace('+', '9').replace('x', '8')
        lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
        transformed_grid = [list(map(int, line)) for line in lines]
        occupancy_grid = np.array(transformed_grid)     

        agent_y_init, agent_x_init = np.where(occupancy_grid == 0) # single value

        agent_y_next, agent_x_next = np.where(occupancy_grid == 8) # multiple values
        agent_y_next = list(agent_y_next)
        agent_x_next = list(agent_x_next)
        agent_next_list = [(x, y) for x, y in zip(agent_x_next, agent_y_next)]

        closest_x_next, closest_y_next = get_closest_gridpoint(point=[agent_x_init[0], agent_y_init[0]], list_of_points= agent_next_list)
        theta_init = np.arctan2(closest_y_next-agent_y_init[0], closest_x_next-agent_x_init[0])

    
        #init_state[f'x{agent}'] = agent_x_init[0]
        #init_state[f'y{agent}'] = agent_y_init[0]
        #init_state[f'theta{agent}'] = theta_init
        init_state.append(float(agent_x_init[0]))
        init_state.append(float(agent_y_init[0]))
        init_state.append(float(theta_init))
        
    return init_state


occupancy_grid_dict = {
            'racetrack_1': """
            ################
            #..............#
            #..............#
            #01............#
            #..............#
            #..............#
            ################""",
            'racetrack_moving_barrier_1':"""
            ################
            #..............#
            #.1............#
            #0.............#
            #..............#
            #..............#
            ################""",
            'racetrack_moving_barrier_2':"""
            ################
            #..............#
            #.1............#
            #0.............#
            #..............#
            #..............#
            ################""",
            'intersection_1': """
            ################
            #####......#####
            #####......#####
            #####......#####
            #####......#####
            #..............#
            #..............#
            #..............#
            #..............#
            #..............#
            #..............#
            #####......#####
            #####......#####
            #####......#####
            #####......#####
            ################""",
            'trichter_1': """
            ################
            #........#######
            #..........#####
            #0.1...........#
            #..........#####
            #........#######
            ################""",
            'door_1_open': """
            #############
            #....########
            #...........#
            #....########
            #############""",
            'door_1_closed': """
            #############
            #....########
            #....#......#
            #....########
            #############""",
            'benchmark_static_dragrace': """
            #######
            #.....#
            #.....#
            #.....#
            #.....#
            #.....#
            #######""",
            'benchmark_dynamic_small1': """
            #########
            #...#...#
            #01.....#
            #...#...#
            #.......#
            #...#...#
            #########""",
            'benchmark_dynamic_small1_1': """
            #########
            #...#...#
            #...#...#
            #...#...#
            #.......#
            #...#...#
            #########""",
            'benchmark_dynamic_small2': """
            ############
            #....##....#
            #...0......#
            #....##....#
            #.1........#
            #....##....#
            ############""",
            'benchmark_dynamic_small2_1': """
            ############
            #....##....#
            #....#.....#
            #....##....#
            #..........#
            #....##....#
            ############""",
            'benchmark_dynamic_small2_2': """
            ############
            #....##....#
            #....#.....#
            #....##....#
            #.....#....#
            #....##....#
            ############""",
            'door_2_open': """
            ###############
            #......########
            #......########
            #.............#
            #......########
            #......########
            ###############""",
            'door_2_closed': """
            ###############
            #......########
            #......########
            #......#......#
            #......########
            #......########
            ###############""",
            'occupancy_grid_maze': """
            ####################
            #...........#......#
            #.....#######......#
            #...........#......#
            #...........#......#
            #.....#............#
            #.....#............#
            #.....#######......#
            #...........#......#
            ####################""",
            'occupancy_grid_free': """
            ....................
            ....................
            ....................
            ....................
            ....................
            ....................
            ....................""",
            'occupancy_grid_10': """
            .......#............
            ....................
            .......#............
            .......#............
            .......#............
            .......#............
            .......#............""",

            'occupancy_grid_11': """
            .......#............
            ....................
            ....................
            .......#............
            .......#............
            .......#............
            .......#............""",

            'occupancy_grid_12': """
            ...#............
            ................
            ...#............
            ...#............
            ...#............
            ...#............
            ...#............""",

            'occupancy_grid_20': """
            .......#............
            ....................
            .......#............
            .......#............
            .......#............
            ....................
            .......#............""",

            'occupancy_grid_21_open': """
            .......#............
            ....................
            .......#............
            .01.................
            .......#............
            ....................
            .......#............""",
            'occupancy_grid_21_closed': """
            .......#............
            .......#............
            .......#............
            ....................
            .......#............
            .......#............
            .......#............""",
            'occupancy_grid_big_open': """
            ##################################################
            #................##..............................#
            #.................##.............................#
            #..................##............................#
            #...................##...........................#
            #....................#...........................#
            #................................................#
            #................................................#
            #....................#...........................#
            #....................#...........................#
            #....................##..........................#
            #.....................##.........................#
            #......................##........................#
            #.......................#........................#
            #.......................#........................#
            #................................................#
            #................................................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            ##################################################""",
            'occupancy_grid_big_closed': """
            ##################################################
            #................##..............................#
            #.................##.............................#
            #..................##............................#
            #...................##...........................#
            #....................#...........................#
            #................................................#
            #................................................#
            #....................#...........................#
            #....................#...........................#
            #....................##..........................#
            #.....................##.........................#
            #......................##........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            #.......................#........................#
            ##################################################""",
            'occupancy_grid_30': """
            ####################
            ..################..
            ....############....
            ....................
            ....############....
            ..################..
            ####################""",
            'occupancy_grid_31': """
            ####################
            .##################.
            .##################.
            ....................
            .##################.
            .##################.
            ####################""",
            'occupancy_grid_40': """
            ....#..
            ...##..
            .......
            ...##..
            ....#..""",
            'occupancy_grid_41': """
            ......#..
            .....##..
            .........
            .........
            .....##..
            ......#..""",

        }