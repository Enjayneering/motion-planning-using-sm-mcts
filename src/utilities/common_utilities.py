import numpy as np

# import modules
import os 
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

# paths
path_to_repository = "/home/enjay/0_thesis/01_MCTS/"
path_to_src = path_to_repository+"src/"

# test file structure 
# including {} for inserting agent indices
path_to_tests = path_to_repository+"tests/"
path_to_test = path_to_tests+"{}/" # insert test_id
path_to_results = path_to_test+"/results/"
path_to_data = path_to_test+"/data/"
path_to_trees = path_to_data+"{}/trees/" # insert agent index
path_to_tree = path_to_trees+"/tree_{}.csv" # insert test_id, agent_index, tree index
path_to_global_state = path_to_data+"global_state.csv"
path_to_rollout_curr = path_to_data+"rollout_curr.csv"
path_to_rollout_last = path_to_data+"rollout_last.csv"
path_to_rollout_tmp = path_to_rollout_last + "~"


# experimental file structure
path_to_experiments = path_to_repository+"experiments/"
new_exp_folder = path_to_experiments+""

#TODO: adjust code

def is_terminal(env, joint_state, curr_timestep, game_timehorizon):
    """
    joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    joint_action: numpy array of shape (n, 2) where n is the number of agents and each row is [v, yaw_rate]
    """
    if curr_timestep >= game_timehorizon:
        #if we reached the maximum timestep
        return True
    elif env.finish_line is not None:
        # if one agent crosses the finish line (x-value, if defined)
        if np.any(joint_state[:,0] >= env.finish_line):
            #print("Terminal state_obj reached")
            return True
    elif env.agent_progress_lines is not None:
        # if one agent reaches a certain progress within the environment
        if any_agent_has_finished(env.agent_progress_lines, joint_state):
            #print("Terminal state_obj reached")
            #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
            return True
    
    else:
        return False

def any_agent_has_finished(env, joint_state, dist_to_end=2):
    # take max progress value from centerline
    joint_positions = joint_state[:,:2] # Get x, y positions

    # Query the KDTree to find the index of the nearest centerline point for each joint position
    for agent_ix, agent_progress_line in env.agent_progress_lines.items():
        max_progress = agent_progress_line['progress'][-int(dist_to_end/env.config_environment['step_dist_progress'])]
        
        distances, indices = agent_progress_line['KDtree'].query(joint_positions)
        # Look up the progress value for the nearest centerline point
        progress = agent_progress_line['progress'][indices]
        if progress >= max_progress:
            return True
    return False

def which_agent_has_finished(env, joint_state, dist_to_end=2):
    # take max progress value from centerline
    joint_positions = joint_state[:,:2] # Get x, y positions

    agents_finished = np.zeros(joint_state.shape[0])

    # Query the KDTree to find the index of the nearest centerline point for each joint position
    for agent_ix, agent_progress_line in env.agent_progress_lines.items():
        dist_ix = int(dist_to_end/env.config_environment['step_dist_progress'])
        max_progress = agent_progress_line['progress'][-dist_ix]
        
        distances, indices = agent_progress_line['KDtree'].query(joint_positions)
        # Look up the progress value for the nearest centerline point
        progress = agent_progress_line['progress'][indices]
        if progress >= max_progress:
            agents_finished[agent_ix] = 1
    return agents_finished

def coll_count(joint_trajectory, config_global):
    """
    joint_trajectory: numpy array of shape (n, 3, k) where n is the number of agents and each row is [x, y, theta] and k is the number of timesteps
    """
    coll_count = 0
    for t in range(joint_trajectory.shape[2] - 1):
        for i in range(joint_trajectory.shape[0]):
            for j in range(i + 1, joint_trajectory.shape[0]):
                line_points_i = np.linspace(joint_trajectory[i, :2, t], joint_trajectory[i, :2, t+1], num=10)
                line_points_j = np.linspace(joint_trajectory[j, :2, t], joint_trajectory[j, :2, t+1], num=10)
                if any(np.linalg.norm(point_i - point_j) <= config_global['collision_distance'] for point_i in line_points_i for point_j in line_points_j):
                    coll_count += 1
    return coll_count

"""def is_collision(state_0, state_1, collision_distance):
    # state = [x, y, theta]
    if distance(state_0, state_1) <= collision_distance:
        return True"""
    
def get_winner(sim):
    final_joint_state = sim.global_trajectory[-1]
    # TODO: adjust for maximum progress
    return which_agent_has_finished(sim.env_global, final_joint_state)
    
def angle_to_goal(state, terminal_state):
    # state [x,y,theta]
    # terminal_state [x,y]
    x_now = state[0]
    y_now = state[1]
    theta_now = state[2]
    x_goal = terminal_state[0]
    y_goal = terminal_state[1]
    angle_to_goal = np.arctan2(y_goal-y_now, x_goal-x_now) - theta_now
    return angle_to_goal

def distance(state_0, state_1):
    # state = [x, y, theta]
    x1 = state_0[0]
    y1 = state_0[1]
    x2 = state_1[0]
    y2 = state_1[1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

"""def get_agent_progress(, prev_state, next_state):
    # state = [x, y, theta]
    # action = [speed, angular_speed]
    prev_closest_coordinate = find_closest_waypoint(Game, prev_state)
    next_closest_coordinate = find_closest_waypoint(Game, next_state)
    progress = next_closest_coordinate[-1] - prev_closest_coordinate[-1]
    return progress"""

def get_joint_env_pos(joint_state, env_centerline):
    """
    joint_state: numpy array of shape (n, s) where n is the number of agents and each row s is a state component like [x, y, theta]
    env_centerline: Dict with each tree and progress list{agent_ix: {KDtree: tree, Progress: list}}
    """
    joint_xy_points = joint_state[:,:2]

    s_coord_vec = np.zeros(joint_xy_points.shape[0])
    for agent_ix in range(joint_xy_points.shape[0]):
        # Query the KDTree for the closest point
        distance, index = env_centerline['KDtree'].query(joint_xy_points)

        # Get the progress value for the closest point
        s_coord = env_centerline['s_coord'][index]
        s_coord_vec[agent_ix] = s_coord
    return s_coord_vec


def get_joint_agent_pos(joint_state, prog_lines):
    """
    joint_state: numpy array of shape (n, s) where n is the number of agents and each row s is a state component like [x, y, theta]
    prog_lines: Dict of Dicts with each tree and progress list{agent_ix: {KDtree: tree, Progress: list}}
    """

    joint_xy_points = joint_state[:,:2]

    s_coord_vec = np.zeros(joint_xy_points.shape[0])
    for agent_ix, agent_progress_line in prog_lines.items():
        # Query the KDTree for the closest point
        distance, index = agent_progress_line['KDtree'].query(joint_xy_points)

        # Get the progress value for the closest point
        s_coord = agent_progress_line['s_coord'][index]
        s_coord_vec[agent_ix] = s_coord
    return s_coord_vec

# Gaussian function
def denormalized_gaussian(x, mu, sigma):
    """works with array of shape (n,1)
    """
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

def coll_count(joint_state, coll_dist=0.5):
    """
    joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    """
    # check all possible agent pairs (a,b)
    coll_vec = np.zeros(len(joint_state))
    for a, agent_a_coord in enumerate(joint_state):
        for b, agent_b_coord in enumerate(joint_state[a+1:]):
            if np.linalg.norm(agent_a_coord, agent_b_coord) < coll_dist:
                # if agents collide
                coll_vec[a] = 1
                coll_vec[b] = 1
    return coll_vec

def roundme(number):
    return number #np.round(number, 2)

def list_data2agent_perspective(lst, index):
    return lst[index:] + lst[:index]

# handling joint actions
def get_max_joint_action(kinematic_params):
    """
    kinematic_params: list of agent kinematic params with each lists of discrete values for each velocity dof
    joint_action: numpy array of shape (n, p) where n is the number of agents and each row is another action parameter
    """
    joint_action = np.zeros((len(kinematic_params), len(kinematic_params[0])))
    for agent_ix, dofs in enumerate(kinematic_params):
        for dof_ix, params in enumerate(dofs):
            joint_action[agent_ix, dof_ix] = max(params)
    return joint_action

def get_all_combis_joint_action(kinematic_params):
    """
    kinematic_params: list of agent kinematic params with each lists of discrete values for each velocity dof
    joint_action: numpy array of shape (n, p, d) where n is the number of agents and each row p is another action parameter and d is the number of different combinations
    
    Example:
    [[[0,1],[-1,0,1]],[[0,1,2],[-2,0,2]]] |[[agent0_action0, agent0_action1],[agent1_action0, agent1_action1, agent1_action2]]
    to:
    array([[[ 0, -1],
        [ 0,  0],
        [ 0,  1],
        [ 1, -1],
        [ 1,  0],
        [ 1,  1]],

       [[ 0, -2],
        [ 0,  0],
        [ 0,  2],
        [ 1, -2],
        [ 1,  0],
        [ 1,  2],
        [ 2, -2],
        [ 2,  0],
        [ 2,  2]]])
    """
    # Create the 3D array
    array_3d = np.array([np.array(np.meshgrid(*params)).T.reshape(-1, len(params)) for agent_params in kinematic_params for params in agent_params])

    # Now, array_3d is a 3D numpy array of shape (n, p, q), where n is the number of agents, p is the number of possible joint action combinations for each agent, and q is the number of action components for each agent.
    #TODO:Check
    return array_3d