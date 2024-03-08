import heapq

def move(loc, dir):
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
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
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


def build_constraint_table(constraints, agent):
    ##############################
    # Task 1.2/1.3: Return a table that constains the list of constraints of
    #               the given agent for each time step. The table can be used
    #               for a more efficient constraint violation check in the 
    #               is_constrained function.

    constraint_table = {'timestep': [], 'agent': [], 'loc': [], 'goal_dist': []}

    for constraint in constraints:
        if agent == constraint['agent']: #only add constraints that are relevant to the agent
            constraint_table['timestep'].append(constraint['timestep'])
            constraint_table['agent'].append(constraint['agent'])
            constraint_table['loc'].append(constraint['loc'])
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
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    ##############################
    # Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
    #               any given constraint. For efficiency the constraints are indexed in a constraint_table
    #               by time step, see build_constraint_table.
    collision = False
    for index, time in enumerate(constraint_table['timestep']):
        if time == next_time:
            if len(constraint_table['loc'][index]) > 1: # edge constraint
                if curr_loc == constraint_table['loc'][index][0] and next_loc == constraint_table['loc'][index][1]: 
                        #print("Theres a collision on the edge of {} and {} at timestep {}".format(curr_loc, next_loc, time))    
                        collision =  True 

            elif next_loc == constraint_table['loc'][index][0]: # vertex constraint
                    #print("Theres a collision caused by agent {} at time {} in cell {}".format(constraint_table['agent'][index], time, next_loc))    
                    collision = True

        if constraint_table['goal_dist'][index] is not None:
            if constraint_table['goal_dist'][index] == 0 and time <= next_time: # constraints after reaching goal
                if next_loc == constraint_table['loc'][index][0]: # vertex constraint
                        #print("Theres a collision caused by agent {} at time {} in cell {}".format(constraint_table['agent'][index], time, next_loc))    
                        collision = True
    return collision

    
    

def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints, max_iter=10**10):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
        max_iter - maximum numbers of iteration until the search stops
    """

    ##############################
    # Task 1.1: Extend the A* search to search in the space-time domain
    #           rather than space domain, only.

    open_list = []
    closed_list = dict() # we index each dictionary by tuples of (cell, time_step)
    earliest_goal_timestep = 10
    h_value = h_values[start_loc]
    iter = 0

    # 1.2: constraints
    constraint_table = build_constraint_table(constraints, agent)

    root = {'loc': start_loc, 'timestep': 0, 'g_val': 0, 'h_val': h_value, 'parent': None}
    push_node(open_list, root)
    closed_list[(root['loc'], root['timestep'])] = root

    while len(open_list) > 0 and iter <= max_iter:
        curr = pop_node(open_list)
        iter += 1
        #############################
        # Task 1.4: Adjust the goal test condition to handle goal constraints
        if curr['loc'] == goal_loc and not any(curr['timestep'] < time_constraint for time_constraint in constraint_table['timestep']):
            return get_path(curr)
        
        for dir in range(4):
            child_loc = move(curr['loc'], dir)
            
            #print("child location new: {}".format(child_loc))
            #print("Shape of map: {}, {}".format(len(my_map), len(my_map[0])))
            #print(my_map)

            if child_loc[0]<0 or child_loc[1]<0 or child_loc[0]>=len(my_map) or child_loc[1]>=len(my_map[0]): # check if new location is on the map
                continue

            if my_map[child_loc[0]][child_loc[1]]: # check if new location is on obstacles
                continue

            # we move in time
            child_move = {'loc': child_loc,
                    'timestep': curr['timestep'] + 1,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
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

            # we rest in time
            child_stay = {'loc': curr['loc'],
                    'timestep': curr['timestep'] + 1,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[curr['loc']],
                    'parent': curr}
            
            if is_constrained(curr['loc'], child_stay['loc'], child_stay['timestep'], constraint_table):
                pass # don't add the child node -> pruning
            
            else:
                if ((child_stay['loc'], child_stay['timestep'])) in closed_list:
                    existing_node = closed_list[(child_stay['loc'], child_stay['timestep'])]
                    if compare_nodes(child_stay, existing_node): # if child is better than existing node, we override it
                        closed_list[(child_stay['loc'], child_stay['timestep'])] = child_stay
                        push_node(open_list, child_stay)
                else:
                    closed_list[(child_stay['loc'], child_stay['timestep'])] = child_stay # if child is not in closed list, it is added
                    push_node(open_list, child_stay) # we have to check for children, so we add it to open list

    return None  # Failed to find solutions
