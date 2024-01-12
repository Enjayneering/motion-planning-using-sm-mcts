import numpy as np

class Environment:
    # define 2D environment with continuous free space and discretized obstacle space
    def __init__(self, config=None):
        self.max_timehorizon = config.max_timehorizon
        self.dynamic_grid = self.init_dynamic_grid(config.env_def)
        self.init_state = self.get_init_state(config.env_def, config.theta_0_init, config.theta_1_init)
        self.occupancy_grid_dict = {
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
    
    def get_init_state(self, env_def, theta_0_init, theta_1_init):
        grid = env_def[0]
        occupancy_grid_define = grid.replace('.', '9').replace('#', '9')
        lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
        transformed_grid = [list(map(int, line)) for line in lines]
        occupancy_grid = np.array(transformed_grid)     

        agent_0_y, agent_0_x = np.where(occupancy_grid == 0)
        agent_1_y, agent_1_x = np.where(occupancy_grid == 1)
        
        init_state = {
            'x0': agent_0_x[0],
            'y0': agent_0_y[0],
            'theta0':theta_0_init,
            'x1': agent_1_x[0],
            'y1': agent_1_y[0],
            'theta1': theta_0_init,
            'timestep': 0,
        }
        return init_state

    def get_occupancy_grid(self, grid):
        occupancy_grid_define = grid.replace('.', '0').replace('0', '0').replace('1', '0').replace('#', '1')
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