from common import *
from kinodynamic_utilities import *
from config import *
import numpy as np



def get_intermediate_payoffs(Game, prev_state_obj, next_state_obj, discount_factor=1):
    #print("prev_state_obj: {}".format(prev_state_obj.get_state_together()))
    #print("next_state_obj: {}".format(next_state_obj.get_state_together()))
    discount = discount_factor**prev_state_obj.timestep

    # DISTANCE BETWEEN AGENTS
    dist_agents = distance(next_state_obj.get_state(agent=0), next_state_obj.get_state(agent=1))

    if dist_agents <= Game.config.collision_distance:
        coll_punishment_0 = np.exp(-dist_agents)
        coll_punishment_1 = np.exp(-dist_agents)
    elif dist_agents > Game.config.collision_distance:
        coll_punishment_0 = 0
        coll_punishment_1 = 0

    if Game.config.collision_ignorance <= 0.5:
        weight_distance_0 = 2*Game.config.collision_ignorance*Game.config.weight_distance
        weight_distance_1 = Game.config.weight_distance
    elif Game.config.collision_ignorance > 0.5:
        weight_distance_0 = Game.config.weight_distance
        weight_distance_1 = 2*(1-Game.config.collision_ignorance)*Game.config.weight_distance
    

    # PROGRESS OF AGENTS
    progress_0 = get_cl_progress(Game, prev_state_obj, next_state_obj, agent=0)
    progress_1 = get_cl_progress(Game, prev_state_obj, next_state_obj, agent=1)

    weight_progress_0 = Game.config.weight_progress
    weight_progress_1 = Game.config.weight_progress

    # NORMALIZATION BOUNDS
    min_coll_0 = 0
    min_coll_1 = 0
    max_coll_0 = np.exp(-0)
    max_coll_1 = np.exp(-0)

    # simple estimates of maximum progress (max velocity * delta_t in negative and positive direction), could be refined based on kinodynamic constraints
    min_progress_0 = -np.max(Game.config.velocity_0)*Game.config.delta_t
    min_progress_1 = -np.max(Game.config.velocity_1)*Game.config.delta_t
    max_progress_0 = np.max(Game.config.velocity_0)*Game.config.delta_t
    max_progress_1 = np.max(Game.config.velocity_1)*Game.config.delta_t

    # NORMALIZED PAYOFF VALUES
    dist_payoff_0 = 1-(coll_punishment_0-min_coll_0)/(max_coll_0-min_coll_0)
    dist_payoff_1 = 1-(coll_punishment_1-min_coll_1)/(max_coll_1-min_coll_1)

    progress_payoff_0 = (progress_0-min_progress_0)/(max_progress_0-min_progress_0)
    progress_payoff_1 = (progress_1-min_progress_1)/(max_progress_1-min_progress_1)

    # WEIGHTING FINAL VALUES
    epsilon = 1e-10  # small constant to prevent division by zero
    intermediate_payoff_0 = (weight_distance_0*dist_payoff_0)/(weight_distance_0+weight_progress_0+epsilon) + (weight_progress_0*progress_payoff_0)/(weight_distance_0+weight_progress_0+epsilon)
    intermediate_payoff_1 = (weight_distance_1*dist_payoff_1)/(weight_distance_1+weight_progress_1+epsilon) + (weight_progress_1*progress_payoff_1)/(weight_distance_1+weight_progress_1+epsilon)
    
    interm_payoffs_sum = discount*[intermediate_payoff_0, intermediate_payoff_1]

    interm_payoffs_each = {'collision_payoff_0': (weight_distance_0/(weight_distance_0+weight_progress_0+epsilon))*discount*dist_payoff_0,
                          'collision_payoff_1': (weight_distance_1/(weight_distance_1+weight_progress_1+epsilon))*discount*dist_payoff_1,
                          'progress_payoff_0': (weight_progress_0/(weight_distance_0+weight_progress_0+epsilon))*discount*progress_payoff_0,
                          'progress_payoff_1': (weight_progress_1/(weight_distance_1+weight_progress_1+epsilon))*discount*progress_payoff_1}

    return interm_payoffs_sum, interm_payoffs_each

def get_final_payoffs(Game, final_state_obj):

    # TIMESTEP 
    if agent_has_finished(Game, final_state_obj, agent=0) and not agent_has_finished(Game, final_state_obj, agent=1): #agent 0 is ahead
        delta_time_0 = Game.config.max_timehorizon-final_state_obj.timestep
        delta_time_1 = 0
    elif not agent_has_finished(Game, final_state_obj, agent=0) and agent_has_finished(Game, final_state_obj, agent=1): #agent 1 is ahead
        delta_time_0 = 0
        delta_time_1 = Game.config.max_timehorizon-final_state_obj.timestep
    else: #draw
        delta_time_0 = 0
        delta_time_1 = 0

    weight_time_0 = Game.config.weight_timestep
    weight_time_1 = Game.config.weight_timestep


    # LEAD OF AGENTS
    progress_0 = find_closest_waypoint(final_state_obj, Game.env.centerlines, agent=0)[-1]
    progress_1 = find_closest_waypoint(final_state_obj, Game.env.centerlines, agent=1)[-1]

    #dist_to_goal_0 = get_cl_progress(Game, final_state_obj, Game.goal_state, agent=0)
    #dist_to_goal_1 = get_cl_progress(Game, final_state_obj, Game.goal_state, agent=1)
    #lead_0 = dist_to_goal_1 - dist_to_goal_0
    #lead_1 = dist_to_goal_0 - dist_to_goal_1

    weight_lead_0 = Game.config.weight_lead
    weight_lead_1 = Game.config.weight_lead

    # AGENT WINNING
    if agent_has_finished(Game, final_state_obj, agent=0) and not agent_has_finished(Game, final_state_obj, agent=1): #agent 0 is ahead
        winning_0 = 1
        winning_1 = 0
    elif not agent_has_finished(Game, final_state_obj, agent=0) and agent_has_finished(Game, final_state_obj, agent=1): #agent 1 is ahead
        winning_0 = 0
        winning_1 = 1
    else: #draw
        winning_0 = 0
        winning_1 = 0

    weight_winning_0 = Game.config.weight_winning
    weight_winning_1 = Game.config.weight_winning

    # NORMALIZATION BOUNDS
    min_time_0 = 0
    min_time_1 = 0
    max_time_0 = Game.config.max_timehorizon
    max_time_1 = Game.config.max_timehorizon

    # lead bounds derived from optimal trajectories percentage advancement
    min_prog_0 = 0
    min_prog_1 = 0
    max_prog_0 = Game.env.centerlines[0][-1][-1]
    max_prog_1 = Game.env.centerlines[1][-1][-1]
    min_lead_0 = -1
    min_lead_1 = -1
    max_lead_0 = 1
    max_lead_1 = 1
   

    min_winning_0 = 0
    min_winning_1 = 0
    max_winning_0 = 1
    max_winning_1 = 1

    # NORMALIZED PAYOFF VALUES
    time_payoff_0 = (delta_time_0-min_time_0)/(max_time_0-min_time_0)
    time_payoff_1 = (delta_time_0-min_time_1)/(max_time_1-min_time_1)

    # lead
    norm_progress_0 = (progress_0-min_prog_0)/(max_prog_0-min_prog_0)
    norm_progress_1 = (progress_1-min_prog_1)/(max_prog_1-min_prog_1)
    lead_0 = norm_progress_0-norm_progress_1
    lead_1 = norm_progress_1-norm_progress_0
    lead_payoff_0 = (lead_0-min_lead_0)/(max_lead_0-min_lead_0)
    lead_payoff_1 = (lead_1-min_lead_1)/(max_lead_1-min_lead_1)

    winning_payoff_0 = (winning_0-min_winning_0)/(max_winning_0-min_winning_0)
    winning_payoff_1 = (winning_1-min_winning_1)/(max_winning_1-min_winning_1)

    # FINAL VALUES
    final_payoff_0 = (weight_time_0*time_payoff_0)/(weight_time_0+weight_lead_0+weight_winning_0) + (weight_lead_0*lead_payoff_0)/(weight_time_0+weight_lead_0+weight_winning_0) + (weight_winning_0*winning_payoff_0)/(weight_time_0+weight_lead_0+weight_winning_0)
    final_payoff_1 = (weight_time_1*time_payoff_1)/(weight_time_1+weight_lead_1+weight_winning_1) + (weight_lead_1*lead_payoff_1)/(weight_time_1+weight_lead_1+weight_winning_1) + (weight_winning_1*winning_payoff_1)/(weight_time_1+weight_lead_1+weight_winning_1)

    #print("Time payoff 0: {}".format(time_payoff_0))
    #print("Time payoff 1: {}".format(time_payoff_1))

    final_values_sum = [final_payoff_0, final_payoff_1]

    final_values_each = {'time_payoff_0': (weight_time_0*time_payoff_0)/(weight_time_0+weight_lead_0+weight_winning_0),
                        'time_payoff_1': (weight_time_1*time_payoff_1)/(weight_time_1+weight_lead_1+weight_winning_1),
                        'lead_payoff_0': (weight_lead_0*lead_payoff_0)/(weight_time_0+weight_lead_0+weight_winning_0),
                        'lead_payoff_1': (weight_lead_1*lead_payoff_1)/(weight_time_1+weight_lead_1+weight_winning_1),
                        'winning_payoff_0': (weight_winning_0*winning_payoff_0)/(weight_time_0+weight_lead_0+weight_winning_0),
                        'winning_payoff_1': (weight_winning_1*winning_payoff_1)/(weight_time_1+weight_lead_1+weight_winning_1)}
    
    return final_values_sum, final_values_each


def get_total_payoffs(Game, interm_payoff_list=None, final_payoff_list=None):
    if len(interm_payoff_list) == 0:
        interm_payoff_list = [[0]*len(Game.Model_params['agents'])]
    if len(final_payoff_list) == 0:
        final_payoff_list = [[0]*len(Game.Model_params['agents'])]

    # NORMALIZING BOUNDS PAYOFFS (each component is normalized already between zero and one)
    min_total_interm = 0
    max_total_interm = len(interm_payoff_list)

    min_total_final = 0
    max_total_final = len(final_payoff_list)

    weight_interm = Game.config.weight_interm
    weight_final = Game.config.weight_final


    total_payoff_list = []
    for agent in Game.Model_params['agents']:
        interm_sum = sum(interm[agent] for interm in interm_payoff_list)

        if max_total_interm == min_total_interm:
            total_payoff_interm = 0
        else:
            total_payoff_interm = (interm_sum-min_total_interm)/(max_total_interm-min_total_interm)

        final_sum = sum(final[agent] for final in final_payoff_list)

        if max_total_final == min_total_final:
            total_payoff_final = 0
        else:
            total_payoff_final = (final_sum-min_total_final)/(max_total_final-min_total_final)

        payoffs_weighted = (weight_interm*total_payoff_interm)/(weight_interm+weight_final) + (weight_final*total_payoff_final)/(weight_interm+weight_final)

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


    