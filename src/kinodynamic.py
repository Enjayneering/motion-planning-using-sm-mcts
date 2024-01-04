import numpy as np
import itertools
from shapely import LineString

from common import *
from environment import *

def mm_unicycle(state, action, delta_t=1):
    # state = [x, y, theta]
    # action = [speed, angular_speed]
    x_new = state[0] + action[0]*np.cos(state[2])*delta_t
    y_new = state[1] + action[0]*np.sin(state[2])*delta_t
    theta_new = (state[2] + action[1]*delta_t)%(2*np.pi) # modulo to keep angle between 0 and 2pi
    return x_new, y_new, theta_new

def is_collision(prev_state, joint_action):
    # check for collision between the agents

    # Create lines for both agents
    state_0 = prev_state.get_state_0()
    action_0 = joint_action[:2]
    x_next_0, y_next_0, theta_next_0 = mm_unicycle(state_0, action_0, Model_params["delta_t"])

    state_1 = prev_state.get_state_1()
    action_1 = joint_action[2:]
    x_next_1, y_next_1, theta_next_1 = mm_unicycle(state_1, action_1, Model_params["delta_t"])

    line_0 = LineString([state_0[:2], [x_next_0, y_next_0]])
    line_1 = LineString([state_1[:2], [x_next_1, y_next_1]])

    if line_0.intersects(line_1):
        print("collision")
        return True
    else:
        print("no collision")
        return False

def is_in_free_space(state, action, init_timestep, num_linesearch = 4):
    x_next, y_next, theta_next = mm_unicycle(state, action)
    dt = Model_params["delta_t"]
    if env.get_current_grid_dict(init_timestep+dt)['x_min'] < x_next < env.get_current_grid_dict(init_timestep+dt)['x_max'] and env.get_current_grid_dict(init_timestep+dt)['y_min'] < y_next < env.get_current_grid_dict(init_timestep+dt)['y_max']:
        # create line inclusing discretized timestep
        line_points = np.linspace(state[:2]+[init_timestep], [x_next, y_next]+[init_timestep+dt], num=num_linesearch).tolist()

        # If any of the points are on the edge of an obstacle, its fine
        #if any(isinstance(num, float) and num % 1 == 0.5 for num in line_points):
        #    return True

        # Sample random points on the line
        #random.sample(line_points, k=80)

        for point in line_points:
            x_point = point[0]
            y_point = point[1]
            time_point = point[2]

            x_round = int(np.round(x_point))
            y_round = int(np.round(y_point))
            time_int = int(time_point) # round down to nearest integer since environment changes at next increment

            # Check if points on obstacles now and in next step
            if env.get_current_grid_dict(time_int)['grid'][y_round, x_round] == 1:
                return False  # Collision detected
    else:
        #print("collision outside grid")
        return False
    #print("no collision")
    return True

def is_in_forbidden_state(state, action, init_timestep, forbidden_states):
    x_next, y_next, theta_next = mm_unicycle(state, action)
    time_next = init_timestep+Model_params["delta_t"]
    if [x_next, y_next, theta_next, time_next] in forbidden_states:
        return True
    else:
        return False


def sample_legal_actions(state_object, forbidden_states=None):
    # representing KINODYNAMIC CONSTRAINTS
    #print("STate object: {}".format(state_object.get_state_together()))
    state_0 = state_object.get_state_0()
    state_1 = state_object.get_state_1()

    # state = [x, y, theta]
    # action = [vel, angular_vel]

    values_speed_0 = Competitive_params['action_set_0']['velocity_0']
    values_speed_1 = Competitive_params['action_set_1']['velocity_1']
    values_angular_speed_0 = Competitive_params['action_set_0']['ang_velocity_0']
    values_angular_speed_1 = Competitive_params['action_set_1']['ang_velocity_1']

    action_tuples_0   = itertools.product(values_speed_0, values_angular_speed_0)
    action_tuples_1   = itertools.product(values_speed_1, values_angular_speed_1)
    sampled_actions_0 = [list(action) for action in action_tuples_0]
    sampled_actions_1 = [list(action) for action in action_tuples_1]

    # prune actions that lead to collision with environment
    sampled_actions_0_pruned = [action_0 for action_0 in sampled_actions_0 if is_in_free_space(state_0, action_0, state_object.timestep)]
    sampled_actions_1_pruned = [action_1 for action_1 in sampled_actions_1 if is_in_free_space(state_1, action_1, state_object.timestep)]

    # prune actions that lead to forbidden states (where the agent cannot get out)
    if forbidden_states:
        sampled_actions_0_pruned = [action_0 for action_0 in sampled_actions_0_pruned if not is_in_forbidden_state(state_0, action_0, state_object.timestep, forbidden_states)]
        sampled_actions_1_pruned = [action_1 for action_1 in sampled_actions_1_pruned if not is_in_forbidden_state(state_1, action_1, state_object.timestep, forbidden_states)]

    # combine both list elements in all possible combinations
    action_pair_pruned = list(itertools.product(sampled_actions_0_pruned, sampled_actions_1_pruned))
    sampled_actions_seperate_pruned = [list(action_pair) for action_pair in action_pair_pruned]
    sampled_actions_together_pruned = [action_pair[0] + action_pair[1] for action_pair in sampled_actions_seperate_pruned]

    #print("sampled_actions_together_pruned: {}".format(sampled_actions_together_pruned))
    return sampled_actions_0_pruned, sampled_actions_1_pruned, sampled_actions_together_pruned