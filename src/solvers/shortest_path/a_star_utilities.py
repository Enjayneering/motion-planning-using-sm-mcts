import numpy as np
import itertools

def mm_unicycle(state, action, delta_t=1):
    # state = [x, y, theta]
    # action = [speed, angular_speed]
    x = round(state[0],0)
    y = round(state[1],0)
    theta = round(state[2],2)
    vel = round(action[0],0)
    ang_vel = round(action[1],2)
    x_new = x + vel*np.cos(theta)*delta_t
    y_new = y + action[0]*np.sin(theta)*delta_t
    theta_new = np.fmod((theta + ang_vel*delta_t), 2*np.pi) # modulo to keep angle between 0 and 2pi
    return round(x_new,0), round(y_new,0), round(theta_new,2)

def sample_legal_actions(env, curr_state, action_set, timestep, delta_t=1):
    # representing KINODYNAMIC CONSTRAINTS
    #print("STate object: {}".format(state_object.get_state_together()))

    # state = [x, y, theta]
    # action = [vel, angular_vel]

    values_speed = action_set['velocity']
    values_angular_speed = action_set['ang_velocity']

    action_tuples   = itertools.product(values_speed, values_angular_speed)
    sampled_actions = [list(action) for action in action_tuples]

    # prune actions that lead to collision with environment
    sampled_actions_pruned = [action for action in sampled_actions if is_in_free_space(env, curr_state, action, timestep, delta_t)]

    return sampled_actions_pruned


def is_in_free_space(env, state, action, init_timestep, delta_t, num_linesearch = 4):
    x_next, y_next, theta_next = mm_unicycle(state, action, delta_t=delta_t)

    if env.get_current_grid(init_timestep+delta_t)['x_min'] <= x_next <= env.get_current_grid(init_timestep+delta_t)['x_max'] and env.get_current_grid(init_timestep+delta_t)['y_min'] <= y_next <= env.get_current_grid(init_timestep+delta_t)['y_max']:
        # create line inclusing discretized timestep
        line_points = np.linspace(list(state)[:2]+[init_timestep], [x_next, y_next]+[init_timestep+delta_t], num=num_linesearch).tolist()

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
            if env.get_current_grid(time_int)['grid'][y_round, x_round] != 0:
                return False  # Collision detected
    else:
        #print("collision outside grid")
        return False
    #print("no collision")
    return True