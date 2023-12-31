from environment import *
from common import *
from kinodynamic import *


"""def get_initial_payoffs(current_state, joint_action): # different since we are using the first step as part of the final trajectory solution
    collision_0 = MCTS_params['penalty_collision_now_0'] if is_collision(current_state, joint_action) else 0

    collision_1 = MCTS_params['penalty_collision_now_1'] if is_collision(current_state, joint_action) else 0

    # payoff at every timestep
    payoff_0 = collision_0
    payoff_1 = collision_1

    return payoff_0, payoff_1"""




def get_collision_penalty(next_state, collision_weight):

    # intermediate reward at timestep k
    dist_agents = distance(next_state.get_state_0()[:2], next_state.get_state_1()[:2])

    collision_0 = -collision_weight*np.exp(-dist_agents*MCTS_params['penalty_collision_0']) if dist_agents < 1 else 0
    collision_1 = -collision_weight*np.exp(-dist_agents*MCTS_params['penalty_collision_1']) if dist_agents < 1 else 0

    # payoff at every timestep
    payoff_0 = collision_0
    payoff_1 = collision_1

    return payoff_0, payoff_1

def get_final_payoffs(final_state):
    # final utility at final timestep k=T
    lead_0 = MCTS_params['reward_lead']*(final_state.x0 - final_state.x1)
    lead_1 = MCTS_params['reward_lead']*(final_state.x1 - final_state.x0)

    y_center = env.dynamic_grid[final_state.timestep]['y_max']/2
    distance_to_center_0 = MCTS_params['penalty_centerline']*np.abs(final_state.y0-y_center)
    distance_to_center_1 = MCTS_params['penalty_centerline']*np.abs(final_state.y1-y_center)

    progress_0 = MCTS_params['reward_progress']*(final_state.x0-env.init_state['x0'])
    progress_1 = MCTS_params['reward_progress']*(final_state.x1-env.init_state['x1'])

    # payoff at final timestep
    payoff_0 = lead_0+distance_to_center_0+progress_0
    payoff_1 = lead_1+distance_to_center_1+progress_1

    return payoff_0, payoff_1

def update_payoff_range(max_payoff, min_payoff, payoff_0, payoff_1):
    if payoff_0 > max_payoff:
        max_payoff = payoff_0
    if payoff_1 > max_payoff:
        max_payoff = payoff_1
    if payoff_0 < min_payoff:
        min_payoff = payoff_0
    if payoff_1 < min_payoff:
        min_payoff = payoff_1
    payoff_range = max_payoff - min_payoff
    return max_payoff, min_payoff, payoff_range

def update_weigths_and_parameters(curr_collision_weight, aver_intermediate_penalties, aver_final_payoff, intermediate_penalties, final_payoff):
    print("max_intermediate_penalties: {}".format(max_intermediate_penalties))
    print("max_final_payoff: {}".format(max_final_payoff))
    # update extremums
    if intermediate_penalties > max_intermediate_penalties:
        max_intermediate_penalties = intermediate_penalties
    if final_payoff > max_final_payoff:
        max_final_payoff = final_payoff
    # update weights
    new_collision_weight = MCTS_params['penalty_collision_delay']*(aver_final_payoff/aver_intermediate_penalties)*curr_collision_weight
    return new_collision_weight, max_intermediate_penalties, max_final_payoff