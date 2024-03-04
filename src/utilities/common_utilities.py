import numpy as np

def is_terminal(env, state, timestep, max_timestep=None):
        # terminal condition
        #print(state_obj.timestep, Game.config.max_timehorizon)
        #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
        if env.finish_line is not None:
            finish_line = env.finish_line
            if state[0] >= finish_line or state[3] >= finish_line:
                #print("Terminal state_obj reached")
                return True
        elif env.centerlines is not None:
            if agent_has_finished(env, state[:3], agent=0) or agent_has_finished(env, state[3:5], agent=1):
                #print("Terminal state_obj reached")
                #print("state_obj {}: {}".format(state_obj.timestep, state_obj.get_state_obj()))
                return True
        if timestep >= max_timestep:
            #print("Max timehorizon reached: {}".format(state_obj.timestep))
            return True
        else:
            return False
        
def agent_has_finished(env, state, agent=0):
    max_progress = env.centerlines[agent][-5][-1] # TODO: specify goal
    if find_closest_waypoint(env.centerlines[agent], state)[-1] >= max_progress:
        return True
    else:
        return False
    
def find_closest_waypoint(centerline, state):
    closest_waypoint = None
    min_distance = float('inf')

    for waypoint in centerline:
        dist = distance(state, waypoint)
        if dist < min_distance:
            min_distance = dist
            closest_waypoint = waypoint
    return closest_waypoint

def distance(state_0, state_1):
    # state = [x, y, theta]
    x1 = state_0[0]
    y1 = state_0[1]
    x2 = state_1[0]
    y2 = state_1[1]
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)