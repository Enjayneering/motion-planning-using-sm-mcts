import random
import os
import numpy as np

current_folder = os.path.dirname(os.path.abspath(__file__))
algo_folder = os.path.dirname(current_folder)
solver_folder = os.path.dirname(algo_folder)
path_to_src = os.path.dirname(solver_folder)
path_to_master = os.path.dirname(path_to_src)

# test file structure
path_to_tests = os.path.join(path_to_master, "tests/")
path_to_test_data = path_to_tests+"data/"
path_to_results = path_to_tests+"results/"
path_to_trees = path_to_test_data+"trees/"
path_to_tree = path_to_trees+"/tree_{}.csv"
path_to_global_state = path_to_test_data+"global_state.csv"
path_to_rollout_curr = path_to_test_data+"rollout_curr.csv"
path_to_rollout_last = path_to_test_data+"rollout_last.csv"
path_to_rollout_tmp = path_to_rollout_last + "~"




freq_stat_data = 10

def is_terminal(Game, state_obj, max_timestep=None):
        # terminal condition
        #print(state_obj.timestep, Game.config.max_timehorizon)
        #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
        if Game.env.finish_line is not None:
            finish_line = Game.env.finish_line
            if state_obj.x0 >= finish_line or state_obj.x1 >= finish_line:
                #print("Terminal state_obj reached")
                return True
        elif Game.env.centerlines is not None:
            if agent_has_finished(Game, state_obj, agent=0) or agent_has_finished(Game, state_obj, agent=1):
                #print("Terminal state_obj reached")
                #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
                return True
        if state_obj.timestep >= max_timestep:
            #print("Max timehorizon reached: {}".format(state_obj.timestep))
            return True
        else:
            return False

def get_max_timehorizon(Game):
    min_time = get_min_time_to_complete(Game)

    max_game_timehorizon = int(Game.config.alpha_terminal * min_time)
    #print("Max game timehorizon: {}".format(max_game_timehorizon))
    return max_game_timehorizon

def get_min_time_to_complete(Game, curr_state=None):
    # state: [x0, y0, theta0, x1, y1, theta1, time]
    min_times = []
    
    if curr_state is None:
            curr_state = Game.init_state

    final_state = Game.terminal_state
    dist_0 = distance(curr_state[0:2], final_state[0:2])
    dist_1 = distance(curr_state[3:5], final_state[3:5])
    #dist_1 = distance(curr_state[3:5], final_state[3:5])
    max_velocity_0 = np.max(Game.config["velocity_0"])
    max_velocity_1 = np.max(Game.config["velocity_1"])
    min_times.append(dist_0/max_velocity_0)
    min_times.append(dist_1/max_velocity_1)    
    return min(min_times)

def coll_count(joint_trajectory):
    coll_count = 0
    for t in range(len(joint_trajectory)-1):
        line_points_0 = np.linspace(joint_trajectory[t][0:2], joint_trajectory[t+1][0:2], num=10).tolist()
        line_points_1 = np.linspace(joint_trajectory[t][3:5], joint_trajectory[t+1][3:5], num=10).tolist()
        if any(distance(point_0, point_1) <= 0.5 for point_0 in line_points_0 for point_1 in line_points_1):
            coll_count += 1
    return coll_count

def agent_has_finished(Game, state_obj, agent=0):
    max_progress = Game.env.centerlines[agent][-1][-1] # TODO: specify goal
    if find_closest_waypoint(Game.env.centerlines[agent], state_obj.get_state(agent=agent))[-1] >= max_progress:
        return True
    else:
        return False
    
def get_winner(Game, state):
    # TODO: adjust for maximum progress
    if agent_has_finished(Game, state, agent=0) and not agent_has_finished(Game, state, agent=1): #agent 0 is ahead
        return [1,0]
    elif not agent_has_finished(Game, state, agent=0) and agent_has_finished(Game, state, agent=1): #agent 1 is ahead
        return [0,1]
    else: #draw
        return [0,0]
    
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

def get_agent_progress(centerline, prev_state, next_state):
    # state = [x, y, theta]
    # action = [speed, angular_speed]
    prev_closest_coordinate = find_closest_waypoint(centerline, prev_state)
    next_closest_coordinate = find_closest_waypoint(centerline, next_state)

    progress = next_closest_coordinate[-1] - prev_closest_coordinate[-1] # extract progress value
    return progress

def get_agent_advancement(centerline, state):
    closest_coordinate = find_closest_waypoint(centerline, state)
    return closest_coordinate[-1]

def find_closest_waypoint(centerline, state):
    closest_waypoint = None
    min_distance = float('inf')

    for waypoint in centerline:
        dist = distance(state, waypoint[:2])
        if dist < min_distance:
            min_distance = dist
            closest_waypoint = waypoint
    return closest_waypoint

def get_env_progress(Game, state):
    if Game.env.env_centerline is not None:
        closest_waypoint = None
        min_distance = float('inf')

        for waypoint in Game.env.env_centerline:
            dist = distance(state, waypoint[:2])
            if dist < min_distance:
                min_distance = dist
                closest_waypoint = waypoint
        return closest_waypoint[-1]
    else:
        return None

# Gaussian function
def denormalized_gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)