import random
import numpy as np

# paths
path_to_repository = "/home/enjay/0_thesis/01_MCTS/"
path_to_src = path_to_repository+"src/"

# test file structure
path_to_tests = path_to_repository+"tests/"
path_to_results = path_to_tests+"results/"
path_to_data = path_to_tests+"data/"
path_to_trees = path_to_data+"trees/"
path_to_tree = path_to_trees+"/tree_{}.csv"
path_to_global_state = path_to_data+"global_state.csv"
path_to_rollout_curr = path_to_data+"rollout_curr.csv"
path_to_rollout_last = path_to_data+"rollout_last.csv"
path_to_rollout_tmp = path_to_rollout_last + "~"


# experimental file structure
path_to_experiments = path_to_repository+"experiments/"
new_exp_folder = path_to_experiments+""

#normalization parameters for UCB
max_payoff = 0
min_payoff = 0
payoff_range = max_payoff - min_payoff

#normalization parameters for payoff weights
aver_intermediate_penalties = 1
aver_final_payoff = 0

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
        elif Game.env.goal_state_obj is not None:
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

    max_game_timehorizon = int(Game.config.alpha_terminal * min_time)+1
    #print("Max game timehorizon: {}".format(max_game_timehorizon))
    return max_game_timehorizon

def get_min_time_to_complete(Game, curr_state=None):
    min_times = []
    
    if curr_state is None:
            curr_state = [Game.env.init_state['x0'], Game.env.init_state['y0'], Game.env.init_state['theta0'],
                          Game.env.init_state['x1'], Game.env.init_state['y1'], Game.env.init_state['theta1']]

    final_state = [Game.env.goal_state['x0'], Game.env.goal_state['y0'], Game.env.goal_state['theta0'],
                   Game.env.goal_state['x1'], Game.env.goal_state['y1'], Game.env.goal_state['theta1']]
    dist_0 = distance(curr_state[0:2], final_state[0:2])
    dist_1 = distance(curr_state[4:6], final_state[4:6])
    #dist_1 = distance(curr_state[3:5], final_state[3:5])
    max_velocity_0 = np.max(Game.config["velocity_0"])
    max_velocity_1 = np.max(Game.config["velocity_1"])
    min_times.append(dist_0/max_velocity_0)
    min_times.append(dist_1/max_velocity_1)    
    return max(min_times)

def coll_count(joint_trajectory):
    coll_count = 0
    for t in range(len(joint_trajectory)-1):
        line_points_0 = np.linspace(joint_trajectory[t][0:2], joint_trajectory[t+1][0:2], num=10).tolist()
        line_points_1 = np.linspace(joint_trajectory[t][3:5], joint_trajectory[t+1][3:5], num=10).tolist()
        if any(distance(point_0, point_1) <= 0.5 for point_0 in line_points_0 for point_1 in line_points_1):
            coll_count += 1
    return coll_count

def agent_has_finished(Game, state_obj, agent=0):
    max_progress = Game.env.env_centerline[-5][-1] # TODO: specify goal
    if find_closest_waypoint(Game, state_obj.get_state(agent=agent))[-1] >= max_progress:
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

def get_agent_progress(Game, prev_state, next_state):
    # state = [x, y, theta]
    # action = [speed, angular_speed]
    prev_closest_coordinate = find_closest_waypoint(Game, prev_state)
    next_closest_coordinate = find_closest_waypoint(Game, next_state)
    progress = next_closest_coordinate[-1] - prev_closest_coordinate[-1]
    return progress

def find_closest_waypoint(Game, state):
    closest_waypoint = None
    min_distance = float('inf')

    for waypoint in Game.env.env_centerline:
        dist = distance(state, waypoint[:2])
        if dist < min_distance:
            min_distance = dist
            closest_waypoint = waypoint
    return closest_waypoint

def get_env_progress(Game, state):
    closest_waypoint = None
    min_distance = float('inf')

    for waypoint in Game.env.env_centerline:
        dist = distance(state, waypoint[:2])
        if dist < min_distance:
            min_distance = dist
            closest_waypoint = waypoint
    return closest_waypoint[-1]

# Gaussian function
def denormalized_gaussian(x, mu, sigma):
    return np.exp(-(x - mu)**2 / (2 * sigma**2))

def manhattan_distance(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)