from .common_utilities import *
from .kinodynamic_utilities import *
#from ....configs.single_experiment_configs.config_utilities import *

import numpy as np

def get_intermediate_payoffs(Game, prev_state_obj, next_state_obj, discount_factor=1):
    #print("prev_state_obj: {}".format(prev_state_obj.get_state_together()))
    #print("next_state_obj: {}".format(next_state_obj.get_state_together()))
    discount = discount_factor**prev_state_obj.timestep
    epsilon = 1e-10  # small constant to prevent division by zero
    
    weights_0 = {}
    payoffs_0 = {}
    weights_1 = {}
    payoffs_1 = {}

    # calculate max progress based on current configuration and maximum speed
    max_next_state_0 = mm_unicycle(prev_state_obj.get_state(agent=0), [np.max(Game.config.velocity_0), np.max(Game.config.ang_velocity_0)])
    max_next_state_1 = mm_unicycle(prev_state_obj.get_state(agent=1), [np.max(Game.config.velocity_1), np.max(Game.config.ang_velocity_1)])

    # DISTANCE BETWEEN AGENTS
    if Game.config.weight_distance > 0:
        dist_agents = distance(next_state_obj.get_state(agent=0), next_state_obj.get_state(agent=1))

        dist_punishment_0 = -denormalized_gaussian(dist_agents, 0, Game.config.collision_distance)
        dist_punishment_1 = -denormalized_gaussian(dist_agents, 0, Game.config.collision_distance)

        # change agents sensibility towards distance
        if Game.config.collision_ignorance <= 0.5:
            weight_distance_0 = 2*Game.config.collision_ignorance*Game.config.weight_distance
            weight_distance_1 = Game.config.weight_distance
        elif Game.config.collision_ignorance > 0.5:
            weight_distance_0 = Game.config.weight_distance
            weight_distance_1 = 2*(1-Game.config.collision_ignorance)*Game.config.weight_distance

        # normalizing bounds

        min_dist_0 = 0
        min_dist_1 = 0
        max_dist_0 = 1
        max_dist_1 = 1

        # for linear distance
        #max_dist_0 = np.sqrt((Game.env.dynamic_grid[0]['x_max']-Game.env.dynamic_grid[0]['x_min'])**2+(Game.env.dynamic_grid[0]['y_max']-Game.env.dynamic_grid[0]['y_min'])**2)
        #max_dist_1 = max_dist_0
        

        # payoff calculation in [0,1]
        dist_payoff_0 = 1-(dist_punishment_0-min_dist_0)/(max_dist_0-min_dist_0)
        dist_payoff_1 = 1-(dist_punishment_1-min_dist_1)/(max_dist_1-min_dist_1)

        weights_0['weight_distance_0'] = weight_distance_0
        payoffs_0['dist_payoff_0'] = dist_payoff_0
        weights_1['weight_distance_1'] = weight_distance_1
        payoffs_1['dist_payoff_1'] = dist_payoff_1
        

    # PROGRESS OF AGENTS
    if Game.config.weight_progress > 0:
        progress_0 = get_agent_progress(Game.env.centerlines[0], prev_state_obj.get_state(agent=0), next_state_obj.get_state(agent=0))
        progress_1 = get_agent_progress(Game.env.centerlines[1], prev_state_obj.get_state(agent=1), next_state_obj.get_state(agent=1))
        
        weight_progress_0 = Game.config.weight_progress
        weight_progress_1 = Game.config.weight_progress

        # normalizing bounds
        max_agent_progress_0 = get_agent_progress(Game.env.centerlines[0], prev_state_obj.get_state(agent=0), max_next_state_0)
        max_agent_progress_1 = get_agent_progress(Game.env.centerlines[1], prev_state_obj.get_state(agent=1), max_next_state_1)
        
        #min_progress_0 = -np.max(Game.config.velocity_0)*Game.config.delta_t
        #min_progress_1 = -np.max(Game.config.velocity_1)*Game.config.delta_t

        # normalizing to [0,1]
        #min_progress_0 = -np.abs(max_agent_progress_0)
        #min_progress_1 = -np.abs(max_agent_progress_0)
        #max_progress_0 = np.abs(max_agent_progress_0)
        #max_progress_1 = np.abs(max_agent_progress_1)
        
        init_env_pos_0 = get_env_progress(Game, prev_state_obj.get_state(agent=0))
        init_env_pos_1 = get_env_progress(Game, prev_state_obj.get_state(agent=1))

        full_env_pos_0 = get_env_progress(Game, max_next_state_0)
        full_env_pos_1 = get_env_progress(Game, max_next_state_1)

        min_progress_0 = 0
        min_progress_1 = 0
        max_progress_0 = full_env_pos_0-init_env_pos_1
        max_progress_1 = full_env_pos_1-init_env_pos_0

        # payoff calculation
        if max_progress_0 > 0:
            progress_payoff_0 = (progress_0-min_progress_0)/(max_progress_0-min_progress_0)
        else:
            progress_payoff_0 = 0
        if max_progress_1 > 0:
            progress_payoff_1 = (progress_1-min_progress_1)/(max_progress_1-min_progress_1)
        else:
            progress_payoff_1 = 0

        weights_0['weight_progress_0'] = weight_progress_0
        payoffs_0['progress_payoff_0'] = progress_payoff_0
        weights_1['weight_progress_1'] = weight_progress_1
        payoffs_1['progress_payoff_1'] = progress_payoff_1


    # PUNISHMENT FOR COLLISION
    if Game.config.weight_collision > 0:
        num_coll_samples = 4
        prev_state_0 = prev_state_obj.get_state(agent=0)
        next_state_0 = next_state_obj.get_state(agent=0)

        # Calculate the start and end points
        start_point_0 = [prev_state_0[0], prev_state_0[1]]
        end_point_0 = [next_state_0[0], next_state_0[1]]

        # Calculate the intermediate points
        intermediate_points_0 = []
        for i in range(1, num_coll_samples):
            intermediate_x = prev_state_0[0] + (next_state_0[0] - prev_state_0[0]) * i / num_coll_samples
            intermediate_y = prev_state_0[1] + (next_state_0[1] - prev_state_0[1]) * i / num_coll_samples
            intermediate_points_0.append([intermediate_x, intermediate_y])
        
        points_0 = [start_point_0] + intermediate_points_0 + [end_point_0]

        prev_state_1 = prev_state_obj.get_state(agent=1)
        next_state_1 = next_state_obj.get_state(agent=1)

        # Calculate the start and end points
        start_point_1 = [prev_state_1[0], prev_state_1[1]]
        end_point_1 = [next_state_1[0], next_state_1[1]]

        # Calculate the intermediate points
        intermediate_points_1 = []
        for i in range(1, num_coll_samples):
            intermediate_x = prev_state_1[0] + (next_state_1[0] - prev_state_1[0]) * i / num_coll_samples
            intermediate_y = prev_state_1[1] + (next_state_1[1] - prev_state_1[1]) * i / num_coll_samples
            intermediate_points_1.append([intermediate_x, intermediate_y])
        
        points_1 = [start_point_1] + intermediate_points_1 + [end_point_1]

        # Check for collision
        for point_0, point_1 in zip(points_0, points_1):
            if distance(point_0, point_1) < Game.config.collision_distance:
                #coll_punishment_0 = -1
                #coll_punishment_1 = -1
                coll_0 = 0
                coll_1 = 0
                break
            else:
                #coll_punishment_0 = 0
                #coll_punishment_1 = 0
                coll_0 = 1
                coll_1 = 1

        weight_collision_0 = Game.config.weight_collision
        weight_collision_1 = Game.config.weight_collision

        # normalizing bounds
        min_coll_0 = 0
        min_coll_1 = 0
        max_coll_0 = 1
        max_coll_1 = 1

        # payoff calculation in [0,1]
        coll_payoff_0 = (coll_0-min_coll_0)/(max_coll_0-min_coll_0)
        coll_payoff_1 = (coll_1-min_coll_1)/(max_coll_1-min_coll_1)

        weights_0['weight_collision_0'] = weight_collision_0
        payoffs_0['coll_payoff_0'] = coll_payoff_0
        weights_1['weight_collision_1'] = weight_collision_1
        payoffs_1['coll_payoff_1'] = coll_payoff_1


    # LEAD PAYOFFS
    if Game.config.weight_lead > 0:
        init_env_pos_0 = get_env_progress(Game, prev_state_obj.get_state(agent=0))
        init_env_pos_1 = get_env_progress(Game, prev_state_obj.get_state(agent=1))

        next_env_pos_0 = get_env_progress(Game, next_state_obj.get_state(agent=0))
        next_env_pos_1 = get_env_progress(Game, next_state_obj.get_state(agent=1))

        full_env_pos_0 = get_env_progress(Game, max_next_state_0)
        full_env_pos_1 = get_env_progress(Game, max_next_state_1)
        
        extr_init_lead_0 = init_env_pos_0-init_env_pos_1
        extr_full_lead_0 = full_env_pos_0-full_env_pos_1
        extr_case1_lead_0 = init_env_pos_0 - full_env_pos_1
        extr_case2_lead_0 = full_env_pos_0 - init_env_pos_1

        extr_init_lead_1 = init_env_pos_1-init_env_pos_0
        extr_full_lead_1 = full_env_pos_1-full_env_pos_0
        extr_case1_lead_1 = init_env_pos_1 - full_env_pos_0
        extr_case2_lead_1 = full_env_pos_1 - init_env_pos_0
        

        min_lead_0 = min(extr_init_lead_0, extr_full_lead_0, extr_case1_lead_0, extr_case2_lead_0)
        min_lead_1 = min(extr_init_lead_1, extr_full_lead_1, extr_case1_lead_1, extr_case2_lead_1)
        max_lead_0 = max(extr_init_lead_0, extr_full_lead_0, extr_case1_lead_0, extr_case2_lead_0)
        max_lead_1 = max(extr_init_lead_1, extr_full_lead_1, extr_case1_lead_1, extr_case2_lead_1)

        lead_0 = next_env_pos_0-next_env_pos_1
        lead_1 = next_env_pos_1-next_env_pos_0

        weight_lead_0 = Game.config.weight_lead
        weight_lead_1 = Game.config.weight_lead

        # calculate payoff

        if (max_lead_0-min_lead_0) > 0:
            lead_payoff_0 = (lead_0-min_lead_0)/(max_lead_0-min_lead_0)
        else:
            lead_payoff_0 = 0
        if (max_lead_1-min_lead_1) > 0:
            lead_payoff_1 = (lead_1-min_lead_1)/(max_lead_1-min_lead_1)
        else:
            lead_payoff_1 = 0

        weights_0['weight_lead_0'] = weight_lead_0
        payoffs_0['lead_payoff_0'] = lead_payoff_0
        weights_1['weight_lead_1'] = weight_lead_1
        payoffs_1['lead_payoff_1'] = lead_payoff_1

    # WEIGHTING VALUES
    total_weight_0 = sum(weights_0.values()) + epsilon
    total_weight_1 = sum(weights_1.values()) + epsilon

    # adding payoffs
    payoffs_0 = {key: value * discount for key, value in payoffs_0.items()}
    payoffs_1 = {key: value * discount for key, value in payoffs_1.items()}

    weighted_payoff_0 = sum(w * p for w, p in zip(weights_0.values(), payoffs_0.values())) / total_weight_0
    weighted_payoff_1 = sum(w * p for w, p in zip(weights_1.values(), payoffs_1.values())) / total_weight_1

    interm_payoffs_sum = [weighted_payoff_0, weighted_payoff_1]
    interm_payoffs_each = {**payoffs_0, **payoffs_1}

    return interm_payoffs_sum, interm_payoffs_each

def get_final_payoffs(Game, final_state_obj, discount_factor=1):
    
    discount = discount_factor**final_state_obj.timestep
    
    # WEIGHTING FINAL VALUES
    epsilon = 1e-10  # small constant to prevent division by zero

    weights_0 = {}
    weights_1 = {}
    payoffs_0 = {}
    payoffs_1 = {}

    # TIMESTEP PAYOFF FOR AGENT WINNING FIRST 
    if Game.config.weight_timestep > 0:
        if agent_has_finished(Game, final_state_obj, agent=0) and not agent_has_finished(Game, final_state_obj, agent=1): #agent 0 is ahead
            delta_time_0 = Game.config.max_timehorizon-final_state_obj.timestep
            delta_time_1 = 0
        elif not agent_has_finished(Game, final_state_obj, agent=0) and agent_has_finished(Game, final_state_obj, agent=1): #agent 1 is ahead
            delta_time_0 = 0
            delta_time_1 = Game.config.max_timehorizon-final_state_obj.timestep
        else: #draw
            delta_time_0 = 0
            delta_time_1 = 0
        #delta_time_0 = Game.config.max_timehorizon-final_state_obj.timestep
        #delta_time_1 = Game.config.max_timehorizon-final_state_obj.timestep

        weight_time_0 = Game.config.weight_timestep
        weight_time_1 = Game.config.weight_timestep

        # normalizing bounds
        min_time_0 = 0
        min_time_1 = 0
        max_time_0 = Game.config.max_timehorizon-get_min_time_to_complete(Game)
        max_time_1 = Game.config.max_timehorizon-get_min_time_to_complete(Game)

        # payoff calculation
        time_payoff_0 = (delta_time_0-min_time_0)/(max_time_0-min_time_0)
        time_payoff_1 = (delta_time_1-min_time_1)/(max_time_1-min_time_1)

        # Add to dictionaries
        weights_0['weight_time_0'] = weight_time_0
        weights_1['weight_time_1'] = weight_time_1
        payoffs_0['final_time_payoff_0'] = time_payoff_0
        payoffs_1['final_time_payoff_1'] = time_payoff_1

    # LEAD OF AGENTS
    if Game.config.weight_final_lead > 0:
        #progress_0 = get_env_progress(Game, final_state_obj.get_state(agent=0))
        #progress_1 = get_env_progress(Game, final_state_obj.get_state(agent=1))
        progress_0 = get_agent_advancement(Game.env.centerlines[0], final_state_obj.get_state(agent=0))
        progress_1 = get_agent_advancement(Game.env.centerlines[1], final_state_obj.get_state(agent=1))
        
        lead_0 = progress_0-progress_1
        lead_1 = progress_1-progress_0

        weight_lead_0 = Game.config.weight_final_lead
        weight_lead_1 = Game.config.weight_final_lead

        # normalizing bounds
        #min_lead_0 = -Game.env.env_centerline[-1][-1]
        #min_lead_1 = -Game.env.env_centerline[-1][-1]
        #max_lead_0 = Game.env.env_centerline[-1][-1]
        #max_lead_1 = Game.env.env_centerline[-1][-1]
        min_lead_0 = -Game.env.centerlines[1][-1][-1]
        min_lead_1 = -Game.env.centerlines[0][-1][-1]
        max_lead_0 = Game.env.centerlines[0][-1][-1]
        max_lead_1 = Game.env.centerlines[1][-1][-1]

        # payoff calculation
        lead_payoff_0 = (lead_0-min_lead_0)/(max_lead_0-min_lead_0)
        lead_payoff_1 = (lead_1-min_lead_1)/(max_lead_1-min_lead_1)

        # Add to dictionaries
        weights_0['weight_lead_0'] = weight_lead_0
        weights_1['weight_lead_1'] = weight_lead_1
        payoffs_0['final_lead_payoff_0'] = lead_payoff_0
        payoffs_1['final_lead_payoff_1'] = lead_payoff_1

    # AGENT WINNING
    if Game.config.weight_winning > 0:
        if agent_has_finished(Game, final_state_obj, agent=0) and not agent_has_finished(Game, final_state_obj, agent=1): #agent 0 is ahead
            winning_0 = 1
            winning_1 = 0
        elif not agent_has_finished(Game, final_state_obj, agent=0) and agent_has_finished(Game, final_state_obj, agent=1): #agent 1 is ahead
            winning_0 = 0
            winning_1 = 1
        else: #draw
            winning_0 = 1
            winning_1 = 1

        weight_winning_0 = Game.config.weight_winning
        weight_winning_1 = Game.config.weight_winning

        # normalizing bounds
        min_winning_0 = 0
        min_winning_1 = 0
        max_winning_0 = 1
        max_winning_1 = 1

        # payoff calculation
        winning_payoff_0 = (winning_0-min_winning_0)/(max_winning_0-min_winning_0)
        winning_payoff_1 = (winning_1-min_winning_1)/(max_winning_1-min_winning_1)

        # Add to dictionaries
        weights_0['weight_winning_0'] = weight_winning_0
        weights_1['weight_winning_1'] = weight_winning_1
        payoffs_0['final_winning_payoff_0'] = winning_payoff_0
        payoffs_1['final_winning_payoff_1'] = winning_payoff_1

    # PROGRESS TO GOAL
    """progress_0 = get_env_progress(Game, final_state_obj, agent=0)
    progress_1 = get_env_progress(Game, final_state_obj, agent=1)

    weight_final_progress_0 = Game.config.weight_final_progress
    weight_final_progress_1 = Game.config.weight_final_progress"""


    """min_progress_0 = 0
    min_progress_1 = 0
    max_progress_0 = Game.env.env_centerline[-1][-1]
    max_progress_1 = Game.env.env_centerline[-1][-1]"""

    """progress_payoff_0 = (progress_0-min_progress_0)/(max_progress_0-min_progress_0)
    progress_payoff_1 = (progress_1-min_progress_1)/(max_progress_1-min_progress_1)"""

    # WEIGHTING VALUES
    total_weight_0 = sum(weights_0.values()) + epsilon
    total_weight_1 = sum(weights_1.values()) + epsilon

    # adding payoffs
    payoffs_0 = {key: value * discount for key, value in payoffs_0.items()}
    payoffs_1 = {key: value * discount for key, value in payoffs_1.items()}

    weighted_payoff_0 = sum(w * p for w, p in zip(weights_0.values(), payoffs_0.values())) / total_weight_0
    weighted_payoff_1 = sum(w * p for w, p in zip(weights_1.values(), payoffs_1.values())) / total_weight_1

    final_payoffs_sum = [weighted_payoff_0, weighted_payoff_1]
    final_payoffs_each = {**payoffs_0, **payoffs_1}

    return final_payoffs_sum, final_payoffs_each


"""    weighted_time_0 = (weight_time_0*time_payoff_0)/(weight_time_0+weight_winning_0+weight_lead_0+epsilon)
    weighted_time_1 = (weight_time_1*time_payoff_1)/(weight_time_1+weight_winning_1+weight_lead_1+epsilon)
    weighted_winning_0 = (weight_winning_0*winning_payoff_0)/(weight_time_0+weight_winning_0+weight_lead_0+epsilon)
    weighted_winning_1 = (weight_winning_1*winning_payoff_1)/(weight_time_1+weight_winning_1+weight_lead_1+epsilon)
    weighted_lead_0 = (weight_lead_0*lead_payoff_0)/(weight_time_0+weight_winning_0+weight_lead_0+epsilon)
    weighted_lead_1 = (weight_lead_1*lead_payoff_1)/(weight_time_1+weight_winning_1+weight_lead_1+epsilon)

    final_payoff_0 = weighted_time_0 + weighted_time_1 + weighted_lead_0
    final_payoff_1 = weighted_winning_0 + weighted_winning_1 + weighted_lead_1
    
    #print("Time payoff 0: {}".format(time_payoff_0))
    #print("Time payoff 1: {}".format(time_payoff_1))

    final_values_sum = [discount*final_payoff_0, discount*final_payoff_1]

    final_values_each = {'time_payoff_0': time_payoff_0,
                        'time_payoff_1': time_payoff_1,
                        'winning_payoff_0': winning_payoff_0,
                        'winning_payoff_1': winning_payoff_1,
                        'lead_payoff_0': lead_payoff_0,
                        'lead_payoff_1': lead_payoff_1}
    return final_values_sum, final_values_each"""


def get_total_payoffs(Game, interm_payoff_list=None, final_payoff_list=None):
    if len(interm_payoff_list) == 0:
        interm_payoff_list = [[0]*len(Game.Model_params['agents'])]
    if len(final_payoff_list) == 0:
        final_payoff_list = [[0]*len(Game.Model_params['agents'])]

    # NORMALIZING BOUNDS PAYOFFS (each component is normalized already between zero and one)
    min_total_interm = 0
    max_total_interm = np.sum([Game.config.discount_factor**k for k in range(len(interm_payoff_list))]) # correct

    min_total_final = 0
    max_total_final = 1

    weight_interm = Game.config.weight_interm
    weight_final = Game.config.weight_final


    total_payoff_list = []
    for agent in Game.Model_params['agents']:
        interm_sum = sum(interm[agent] for interm in interm_payoff_list)

        total_payoff_interm = (interm_sum-min_total_interm)/(max_total_interm-min_total_interm)

        final_sum = sum(final[agent] for final in final_payoff_list)

        total_payoff_final = (final_sum-min_total_final)/(max_total_final-min_total_final)

        # WEIGHT TOTAL PAYOFF
        epsilon = 1e-10  # small constant to prevent division by zero
        payoffs_weighted = (weight_interm*total_payoff_interm)/(weight_interm+weight_final+epsilon) + (weight_final*total_payoff_final)/(weight_interm+weight_final+epsilon)

        total_payoff_list.append(payoffs_weighted)

    return total_payoff_list


def get_total_payoffs_cfr(Game, terminal_node):

    state_traj = []
    node = terminal_node
    while node.parent is not None:
        state_traj.append(node.state)
        node = node.parent
    state_traj.reverse()

    interm_payoffs = []
    final_payoffs = []
    for i in range(len(state_traj)-1):
        interm_payoff_sum, _ = get_intermediate_payoffs(Game, state_traj[i], state_traj[i+1])
        interm_payoffs.append(interm_payoff_sum)

    final_payoff_sum, _ = get_final_payoffs(Game, state_traj[-1])
    final_payoffs.append(final_payoff_sum)

    return get_total_payoffs(Game, interm_payoff_list=interm_payoffs, final_payoff_list=final_payoffs)


    