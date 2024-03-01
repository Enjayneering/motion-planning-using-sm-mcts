import itertools
from shapely import LineString
import random
import numpy as np

# import modules
import os 
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

from common_utilities import *

#TODO: DONE

"""def mm_unicycle(joint_state, joint_action, delta_t=1):
    # state = [x, y, theta, timestep,...]
    # action = [speed, angular_speed,...]
    state = np.array(joint_state).reshape(-1, 1) #reshape state list to column vector
    action = np.array(joint_action).reshape(-1, 1) #reshape action list to column vector

    


    x_new = state[0] + action[0]*np.cos(state[2])*delta_t
    y_new = state[1] + action[0]*np.sin(state[2])*delta_t
    theta_new = np.fmod((state[2] + action[1]*delta_t), 2*np.pi) # modulo to keep angle between 0 and 2pi
    timestep_new = state[3] + delta_t
    return [x_new, y_new, theta_new, timestep_new]"""

def joint_kinematic_bicycle_model(joint_state, joint_action, delta_t=1, L=1.0):
    """
    joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    joint_action: numpy array of shape (n, 2) where n is the number of agents and each row is [v, yaw_rate]
    delta_t: time step size
    L: wheelbase of the bicycle model
    """
    # Extract x, y, theta from the joint_state
    x = joint_state[:, 0]
    y = joint_state[:, 1]
    theta = joint_state[:, 2]

    # Extract v (velocity) and yaw_rate from the joint_action
    v = joint_action[:, 0]
    yaw_rate = joint_action[:, 1]

    # Compute the new joint_state using the kinematic bicycle model
    x_new = x + v * np.cos(theta) * delta_t
    y_new = y + v * np.sin(theta) * delta_t
    theta_new = theta + (v / L) * np.tan(yaw_rate) * delta_t

    # Combine x_new, y_new, theta_new into a new joint_state array
    joint_state_new = np.vstack((x_new, y_new, theta_new)).T

    return joint_state_new

def single_kinematic_bicycle_model(state, action, delta_t=1, L=1.0):
    """
    state: numpy array of shape (3,) where each element is [x, y, theta]
    action: numpy array of shape (2,) where each element is [v, yaw_rate]
    delta_t: time step size
    L: wheelbase of the bicycle model
    """
    # Extract x, y, theta from the joint_state
    x = state[0]
    y = state[1]
    theta = state[2]

    # Extract v (velocity) and yaw_rate from the joint_action
    v = action[0]
    yaw_rate = action[1]

    # Compute the new joint_state using the kinematic bicycle model
    x_new = x + v * np.cos(theta) * delta_t
    y_new = y + v * np.sin(theta) * delta_t
    theta_new = theta + (v / L) * np.tan(yaw_rate) * delta_t

    # Combine x_new, y_new, theta_new into a new joint_state array
    state_new = np.array([x_new, y_new, theta_new])

    return state_new


def is_collision(state_0, state_1, collision_distance):
    # state = [x, y, theta]
    if np.linalg.norm(state_0[:2], state_1[:2]) <= collision_distance:
        return True

"""def lines_intersect(Game, prev_state, joint_action):
    # check for collision between the agents

    # Create lines for both agents
    state_0 = prev_state.get_state(agent=0)
    action_0 = joint_action[:2]
    x_next_0, y_next_0, theta_next_0 = mm_unicycle(state_0, action_0, delta_t=Game.Model_params["delta_t"])

    state_1 = prev_state.get_state(agent=1)
    action_1 = joint_action[2:]
    x_next_1, y_next_1, theta_next_1 = mm_unicycle(state_1, action_1, delta_t=Game.Model_params["delta_t"])

    line_0 = LineString([state_0[:2], [x_next_0, y_next_0]])
    line_1 = LineString([state_1[:2], [x_next_1, y_next_1]])

    if line_0.intersects(line_1):
        print("collision")
        return True
    else:
        print("no collision")
        return False"""

def single_move_is_forbidden(state, action, curr_timestep, forbidden_states, delta_t=1):
    x_next, y_next, theta_next = single_kinematic_bicycle_model(state, action, delta_t=delta_t)
    time_next = curr_timestep+delta_t
    if [x_next, y_next, theta_next] in forbidden_states[time_next]:
        return True
    else:
        return False
    
def single_move_is_in_free_space(env, state, action, curr_timestep, delta_t=1, num_linesearch=4):
    x_next, y_next, theta_next = single_kinematic_bicycle_model(state, action, delta_t=delta_t)
    
    # check if next state is within the grid
    current_grid = env.get_current_grid(curr_timestep)
    x_min, x_max = current_grid['x_min'], current_grid['x_max']
    y_min, y_max = current_grid['y_min'], current_grid['y_max']

    if x_min <= x_next <= x_max and y_min <= y_next <= y_max:
        # create line including discretized timestep
        line_points = np.linspace(state[:2]+[curr_timestep], [x_next, y_next]+[curr_timestep+delta_t], num=num_linesearch)

        x_points = np.round(line_points[:, 0]).astype(int)
        y_points = np.round(line_points[:, 1]).astype(int)
        time_points = line_points[:, 2].astype(int)

        # Check if points on obstacles now and in next step
        if np.any(current_grid['grid'][y_points, x_points] != 0):
            return False  # Collision detected
    else:
        return False  # Collision outside grid
    return True  # No collision

def joint_sample_legal_actions(sim, joint_state, kinematic_params, curr_timestep, delta_t=1):
    """
    joint_state: list of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    joint_action: list of shape (n, 2) where n is the number of agents and each row is [v, yaw_rate]
    tensor storing all joint action combinations: (c, n, a) where c is the number of combinations, n is the number of agents and a is the number of actions
    note: if an action for a agent is not feasible, the whole joint action is not feasible
    """
    # Get all combinations of joint actions
    action_combinations = list(itertools.product(*kinematic_params))

    # Create a filter list of the same size as action_combinations
    joint_action_filter = [True] * len(action_combinations)

    # Filter and delete all actions that are not feasible for each agent
    for joint_action_ix, joint_action in enumerate(action_combinations):
        for agent_ix, (agent_state, agent_action) in enumerate(zip(joint_state, joint_action)):
            # Check if the action is feasible for the agent
            if single_move_is_forbidden(agent_state, agent_action, curr_timestep, sim.forbidden_states, delta_t) or \
            not single_move_is_in_free_space(sim, agent_state, agent_action, curr_timestep, delta_t):
                # If the action is not feasible, set the corresponding element in the filter list to False
                joint_action_filter[joint_action_ix] = False
                break

    # Filter out all actions that are not feasible
    action_combinations = [action for action, is_feasible in zip(action_combinations, joint_action_filter) if is_feasible]

    return action_combinations

    """# Get all combinations of joint actions
    action_combinations = list(itertools.product(*kinematic_params))

    # Convert list of tuples to numpy array and reshape it to add a third dimension (combinations, agents, actions)
    action_combinations_array = np.array(action_combinations).reshape(len(kinematic_params), -1, len(kinematic_params[0]))

    # Create a filter array of the same size as action_combinations_array
    joint_action_filter = np.ones(action_combinations_array.shape[0], dtype=bool)

    # filter and delete all actions that are not feasible for each agent
    for joint_action_ix, joint_action in enumerate(action_combinations_array):
        for agent_ix, (agent_state, agent_action) in enumerate(zip(joint_state, joint_action)):
            # Check if the action is feasible for the agent
            if single_move_is_forbidden(agent_state, agent_action, curr_timestep, sim.forbidden_states, delta_t) or \
            not single_move_is_in_free_space(sim, agent_state, agent_action, curr_timestep, delta_t):
                # If the action is not feasible, set the corresponding element in the filter array to False
                joint_action_filter[joint_action_ix] = False
                break
    # filter out all actions that are not feasible
    action_combinations_array = action_combinations_array[joint_action_filter]
    
    return action_combinations_array"""
