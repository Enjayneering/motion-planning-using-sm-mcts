import numpy as np

from superparameter import *

class Environment:
    # define 2D environment with continuous free space and discretized obstacle space
    def __init__(self):
        self.env_name_trigger = [(0,'occupancy_grid_21_open'), (3, 'occupancy_grid_21_closed')]
        self.init_state = {
            'x0': 1,
            'y0': 1,
            'theta0': 0,
            'x1': 1,
            'y1': 5,
            'theta1': 0,
            'timestep': 0,
        }
        self.occupancy_grid_dict = {
            'box': """
            ####
            #..#
            #..#
            ####""",
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
            ....................
            .......#............
            ....................
            .......#............""",
            'occupancy_grid_21_closed': """
            .......#............
            ....................
            .......#............
            ....................
            .......#............
            .......#............
            .......#............""",
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
        self.dynamic_grid = self.get_dynamic_grid()


    def get_static_grid(self, choice):
        grid = self.occupancy_grid_dict[choice]

        occupancy_grid_define = grid.replace('.', '0').replace('#', '1')
        lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
        transformed_grid = [list(map(int, line)) for line in lines]

        occupancy_grid = np.array(transformed_grid)
        return occupancy_grid

    def get_dynamic_grid(self):
        env_list = []
        i = 0

        for timestep in range(game_horizon+3): #TODO: check index error
            if i < len(self.env_name_trigger) and timestep == self.env_name_trigger[i][0]:
                env_list.append({'grid': self.get_static_grid(self.env_name_trigger[i][1]),
                                 'x_min': self.get_x_min(self.env_name_trigger[i][1]),
                                 'y_min': self.get_y_min(self.env_name_trigger[i][1]),
                                 'x_max': self.get_x_max(self.env_name_trigger[i][1]),
                                 'y_max': self.get_y_max(self.env_name_trigger[i][1]),})
                i += 1
            else:
                env_list.append((env_list[-1]))  # fill with previous environment

        return env_list

    def get_x_min(self, env_name):
            static_grid = self.get_static_grid(env_name)
            min_x = 0

            for column in range(static_grid.shape[1]):
                if any(static_grid[:,min_x] == 0):
                    #print("min progress: {}".format(min_x))
                    return min_x
                else:
                    min_x += 1

    def get_y_min(self, env_name):
        static_grid = self.get_static_grid(env_name)
        min_y = 0

        for row in range(static_grid.shape[0]):
            if any(static_grid[min_y,:] == 0):
                #print("min progress: {}".format(min_y))
                return min_y
            else:
                min_y += 1

    def get_x_max(self, env_name):
        static_grid = self.get_static_grid(env_name)
        max_x = static_grid.shape[1]-1

        for column in range(max_x):
            if any(static_grid[:,max_x] == 0):
                #print("max progress: {}".format(max_x))
                return max_x
            else:
                max_x -= 1

    def get_y_max(self, env_name):
        static_grid = self.get_static_grid(env_name)
        max_y = static_grid.shape[0]-1

        for row in range(max_y):
            if any(static_grid[max_y,:] == 0):
                #print("max progress: {}".format(max_y))
                return max_y
            else:
                max_y -= 1




# create environment
env = Environment()