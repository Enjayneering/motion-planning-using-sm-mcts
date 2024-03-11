import heapq
import numpy as np
from scipy.stats import norm
from utilities.common_utilities import *

from a_star_utilities import *

#loc = (x, y, theta)

def move(loc, dir):
    # move to neighboring cells with unit timesteps
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def get_sum_of_cost(paths):
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst


def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= my_map.shape[1] \
               or child_loc[1] < 0 or child_loc[1] >= my_map.shape[0]:
               continue
            if my_map[child_loc[1]][child_loc[0]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def build_constraint_table(constraints):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    constraint_table = {'timestep': [], 'loc': [], 'goal_dist': []}

    for constraint in constraints if constraints is not None else []:
        constraint_table['timestep'].append(constraint['timestep'])
        constraint_table['loc'].append(constraint['loc'][:2])
        if constraint['goal_dist'] is not None:
            constraint_table['goal_dist'].append(constraint['goal_dist'])
    #print("constraint_table: {}".format(constraint_table))
    return constraint_table


def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location


def get_path(goal_node):
    action_path = []
    state_path = []
    time_path = []
    curr = goal_node
    while curr['parent_action'] is not None:
        action_path.append(curr['parent_action'])
        state_path.append(curr['loc'])
        time_path.append(curr['timestep'])
        curr = curr['parent']
    state_path.reverse()
    action_path.reverse()
    time_path.reverse()
    return action_path, state_path, time_path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    collision = False
    curr_loc = curr_loc[:2]
    next_loc = next_loc[:2]
    for index, time in enumerate(constraint_table['timestep']):
        if next_time == time:
            """if len(constraint_table['loc'][index]) > 1: # edge constraint
                if curr_loc == constraint_table['loc'][index] and next_loc == constraint_table['loc'][index]: 
                        #print("Theres a collision on the edge of {} and {} at timestep {}".format(curr_loc, next_loc, time))    
                        collision =  True """

            if next_loc == constraint_table['loc'][index]: # vertex constraint
                    #print("Theres a collision caused by agent {} at time {} in cell {}".format(constraint_table['agent'][index], time, next_loc))    
                    collision = True
                    break

        """if constraint_table['goal_dist'][index] is not None:
            if constraint_table['goal_dist'][index] == 0 and time <= next_time: # constraints after reaching goal
                if next_loc == constraint_table['loc'][index]: # vertex constraint
                        #print("Theres a collision caused by agent {} at time {} in cell {}".format(constraint_table['agent'][index], time, next_loc))    
                        collision = True"""
    return collision


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'] + node['d_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] + n1['d_val'] < n2['g_val'] + n2['h_val'] + n2['d_val']
    # only judje by number of timesteps taken
    #return n1['g_val'] < n2['g_val']


def a_star(env, start_loc, goal_loc, h_values, constraints=None, max_iter=10**10, action_set=None, delta_t=1, start_timestep=0, scale_dense=1):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
        max_iter - maximum numbers of iteration until the search stops
    """
    density_map = generate_density_map(env.get_current_grid(start_timestep)['grid'].shape, start_loc[:2], goal_loc[:2], scale=scale_dense)
    #density_map = generate_random_density_map(env.get_current_grid(start_timestep)['grid'].shape, scale=scale_dense)
    
    trajectories = {'actions': [], 'states': [], 'times': []}
    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict() # we index each dictionary by tuples of (cell, time_step)
    earliest_goal_timestep = 10
    
    h_value = h_values[start_loc[:2]]
    iter = 0
    num_traj = 0

    # 1.2: constraints
    constraint_table = build_constraint_table(constraints)

    
 
    root = {'loc': start_loc, 'timestep': start_timestep, 'g_val': 0, 'h_val': h_value, 'd_val': density_map[start_loc[:2]], 'parent_action': None, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root

    while len(open_list) > 0 and iter <= max_iter:
        curr = pop_node(open_list)
        iter += 1
        #############################
        
        # Adjust so that we collect best trajectories
        if curr['loc'] == goal_loc and not any(curr['timestep'] < time_constraint for time_constraint in constraint_table['timestep']):
            action_path, state_path, time_path = get_path(curr)
            print('Trajectory found: ', action_path)
            return action_path, state_path, time_path
        
        legal_actions = sample_legal_actions(env, curr['loc'], action_set, curr['timestep'], delta_t)
        for action in legal_actions:
            #child_loc = move(curr['loc'], dir)
            child_loc = mm_unicycle(curr['loc'], action, delta_t=delta_t)
            
            child_loc_hval =  tuple(round(x) for x in child_loc[:2])
            
            #print("child location new: {}".format(child_loc))
            #print("Shape of map: {}, {}".format(len(my_map), len(my_map[0])))
            #print(my_map)

            """if child_loc[0]<0 or child_loc[1]<0 or child_loc[0]>=len(my_map) or child_loc[1]>=len(my_map[0]): # check if new location is on the map
                continue

            if my_map[child_loc[0]][child_loc[1]]: # check if new location is on obstacles
                continue"""

            # we apply action in time
            child_move = {'loc': child_loc,
                    'timestep': curr['timestep'] + delta_t,
                    'g_val': curr['g_val'] + 1, # g_val indicates a decision timestep
                    'h_val': h_values[child_loc_hval],
                    'd_val': density_map[child_loc_hval],
                    'parent_action': action,
                    'parent': curr}
            
            if is_constrained(curr['loc'], child_move['loc'], child_move['timestep'], constraint_table):
                pass # don't add the child node -> pruning

            else:
                if ((child_move['loc'], child_move['timestep'])) in closed_list:
                    existing_node = closed_list[(child_move['loc'], child_move['timestep'])]
                    if compare_nodes(child_move, existing_node): # if child is better than existing node, we override it
                        closed_list[(child_move['loc'], child_move['timestep'])] = child_move
                        push_node(open_list, child_move)
                else:
                    closed_list[(child_move['loc'], child_move['timestep'])] = child_move # if child is not in closed list, it is added
                    push_node(open_list, child_move) # we have to check for children, so we add it to open list
            
    return None  # Failed to find solutions



def line_equation(x1, y1, x2, y2):
    """Calculate the slope (m) and y-intercept (c) of the line."""
    m = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
    c = y1 - m * x1
    return m, c

def perpendicular_distance(x, y, m, c):
    """Calculate perpendicular distance from point (x, y) to the line y = mx + c."""
    if m == np.inf:
        return 0  # Vertical line case
    return abs(-m * x + y - c) / np.sqrt(m**2 + 1)

def generate_density_map(size, start, goal, scale=1):
    """
    Generates a density map that encourages paths diverging perpendicularly from the straight path.
    
    Parameters:
        size: Tuple of (height, width) for the density map size.
        start, goal: Start and goal points as (x, y) tuples.
        scale: Scaling factor for density values.
    """
    height, width = size
    m, c = line_equation(start[0], start[1], goal[0], goal[1])
    density_map = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            distance = perpendicular_distance(x, y, m, c)
            density_map[y, x] = denormalized_gaussian(distance, 0, 1)*scale

    # Normalize or scale density values as needed
    """max_density = np.max(density_map)
    if max_density > 0:
        density_map = density_map / max_density * scale"""

    # change dimensions to match x, y axis
    return np.transpose(density_map)

def generate_random_density_map(size, scale=1):
    """
    Generates a random density map.
    
    Parameters:
        size: Tuple of (height, width) for the density map size.
        scale: Scaling factor for density values.
    """
    height, width = size
    density_map = np.random.rand(height, width) * scale

    # change dimensions to match x, y axis
    return np.transpose(density_map)