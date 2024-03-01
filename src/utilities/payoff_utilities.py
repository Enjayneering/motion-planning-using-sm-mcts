import numpy as np
from scipy.interpolate import interp1d

# import modules
import os 
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

from common_utilities import *
from kinodynamic_utilities import *

#TODO: Change to each agent

def get_total_payoff(final_trajectory, env, config_payoffs, kinematic_params, delta_t, discount_factor=1, epsilon=1e-10):
    """
    trajectory: list of joint_states (from initial to final state)
    joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    payoff_matrix: numpy array of shape (n,p) where n is the number of agents and each row p is a certain payoff
    weight_matrix: numpy array of shape (n,p) where n is the number of agents and each row p is a certain weight (assuming all agents share the same weights)

    """
    payoff_data_log = {} # storing timestep: payoff_matrix

    # import weights from config
    conf_weights_interm = config_payoffs['weight_interm_vec']
    conf_weights_final = config_payoffs['weight_final_vec']
    conf_weight_interm_glob = config_payoffs['weight_interm']
    conf_weight_final_glob = config_payoffs['weight_final']
    
    coll_dist = config_payoffs['collision_distance']

    # characteristic values
    n_agents = len(final_trajectory['states'][0])
    n_timesteps = len(final_trajectory['timesteps'])
    n_weights_interm = len(conf_weights_interm)
    n_weights_final = len(conf_weights_final)

    # initialize payoff weight matrices (for all agents and payoffs)
    weights_interm_matrix = np.zeros((n_agents, n_weights_interm))
    for ix, weight in enumerate(conf_weights_interm.values()):
        weights_interm_matrix[:, ix] = weight

    weights_final_matrix = np.zeros((n_agents, n_weights_final))
    for ix, weight in enumerate(conf_weights_final.values()):
        weights_final_matrix[:, ix] = weight
    
    # organize payoff functions
    payoff_interm_func = {"dist": get_payoff_dist,
                        "coll": get_payoff_coll,
                        "prog": get_payoff_prog,
                        "comp": get_payoff_comp,}
    
    payoff_final_func = {"time": get_payoff_time,
                        "win": get_payoff_win,
                        "lead": get_payoff_lead,}
    
    #####################
    # Main Calculations #
    #####################

    interm_payoffs_matrix_glob = get_interm_payoffs(kinematic_params)
    final_payoffs_matrix_glob = get_final_payoffs(kinematic_params)

    # get total numpy array of shape (n, p) where n is the number of agents and each row p is a certain payoff
    total_payoffs_weighted = (conf_weight_interm_glob*interm_payoffs_matrix_glob + conf_weight_final_glob*final_payoffs_matrix_glob)/(conf_weight_interm_glob+conf_weight_final_glob+epsilon)

    # reduce it to a (n,1) array representing the total payoff for each agent
    total_payoffs_vec_glob = np.mean(total_payoffs_weighted, axis=1)

    ####################################
    # Functions for payoff calculation #
    ####################################

    def get_interm_payoffs(kinematic_params):        
        #### Compute intermediate payoffs
        interm_payoff_matrix = np.zeros((n_agents, n_weights_interm, n_timesteps))
        for k in final_trajectory['timesteps'][:-1]:
            #################
            # charac values #
            #################
            curr_joint_state = final_trajectory['states'][k]
            next_joint_state = final_trajectory['states'][k+1]

            max_joint_action = get_max_joint_action(kinematic_params)
            max_next_state = joint_kinematic_bicycle_model(curr_joint_state, max_joint_action, delta_t=delta_t)
            discount = discount_factor**k
            
            # calculate all payoff components at timestep k

            curr_payoff_matrix = np.zeros((n_agents, n_weights_interm))
            for payoff_ix, payoff_key in enumerate(conf_weights_interm.keys()):
                # get the payoff component in shape (n, 1)

                payoff_component = call_payoff_func(payoff_interm_func[payoff_key], curr_joint_state=curr_joint_state, next_joint_state=next_joint_state, env=env, coll_dist=coll_dist, max_next_state=max_next_state, config_agent=config_agent, config_global=config_global, discount_factor=discount_factor)
                # add the component to payoff matrix at timestep k
                curr_payoff_matrix[:, payoff_ix] = payoff_component
                
            # weight the payoff components
            weighted_payoff_matrix_timestep_k = discount*get_weighted_payoff_matrix(curr_payoff_matrix, weights_interm_matrix, epsilon=epsilon)
            
            interm_payoff_matrix[:, :, k] = weighted_payoff_matrix_timestep_k
           
            # save payoff data for logging
            payoff_data_log[k] = weighted_payoff_matrix_timestep_k

        # take mean over all timesteps
        interm_payoff_matrix = np.mean(interm_payoff_matrix, axis=2)

        return interm_payoff_matrix
    
    def get_final_payoffs(kinematic_params):
        #### Compute final payoffs        
        discount = discount_factor**final_timestep
        
        final_joint_state = final_trajectory['states'][-1]
        final_timestep = final_trajectory['timesteps'][-1]

        # calculate all final payoffs
        final_payoff_matrix = np.zeros((n_agents, n_weights_final))
        for payoff_ix, payoff_key in enumerate(conf_weights_final.keys()):
            # get the payoff component in shape (n, 1)
            payoff_component = payoff_final_func[payoff_key](final_joint_state)
            # add the component to payoff matrix at timestep k
            final_payoff_matrix[:, payoff_ix] = payoff_component
        
        # weight the payoff components
        weighted_final_payoff_matrix = discount*get_weighted_payoff_matrix(final_payoff_matrix, weights_final_matrix, epsilon=epsilon)
        # save payoff data for logging
        payoff_data_log[final_timestep] = weighted_final_payoff_matrix

        return weighted_final_payoff_matrix

    return total_payoffs_vec_glob, payoff_data_log

def get_weighted_payoff_matrix(payoff_matrix, weight_matrix, epsilon=1e-10):
    """
    payoff_matrix: numpy array of shape (n, p) where n is the number of agents and each row p is a certain payoff
    weight_matrix: numpy array of shape (n, p) where n is the number of agents and each row p is a certain weight
    returns: weighted payoff array of shape (n, p) with rescaled payoffs
    """
    weighted_payoff_matrix = np.zeros(payoff_matrix.shape)
    for agent_ix in range(payoff_matrix.shape[0]):
        weight_vec = weight_matrix[agent_ix, :]
        payoff_vec = payoff_matrix[agent_ix, :]
        weighted_payoff = np.dot(weight_vec, payoff_vec)/(np.sum(weight_vec)+epsilon)
        weighted_payoff_matrix[agent_ix, :] = weighted_payoff
    return weighted_payoff_matrix


#################################
# Payoff component calculations #
#################################
def call_payoff_func(func, *args, **kwargs):
    return func(*args, **kwargs)

def get_payoff_dist(next_joint_state, coll_dist):
    """
    joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    returns: payoff array of shape (n, 1)
    """
    # Calculate the distance between the agents
    dist_agents = np.linalg.norm(next_joint_state[:, :2] - next_joint_state[:, :2])

    # Calculate the distance punishment
    means = np.zeros(len(dist_agents))
    variances = coll_dist * np.ones(len(dist_agents))
    dist_punishment = -denormalized_gaussian(dist_agents, means, variances)

    min_dist = np.zeros(len(dist_agents))
    max_dist = np.ones(len(dist_agents))

    # Calculate the distance payoff
    payoff_dist = 1-(dist_punishment-min_dist)/(max_dist-min_dist)

    return payoff_dist

def get_payoff_coll(curr_joint_state, next_joint_state, coll_dist):
    """
    joint_state: numpy array of shape (n, 3) where n is the number of agents and each row is [x, y, theta]
    returns: payoff array of shape (n, 1)
    """
    # interpolate between the start and end state (only requires x,y coordinates) < array of size (d,n,s) with interp. depth d, n agents, and s states
    interp_states = interpolate_joint_states(curr_joint_state[:, :2], next_joint_state[:, :2], n_interp=2)
    coll_vec = search_collisions_interp(interp_states, coll_dist=coll_dist)
    # transform coll_vec to payoff
    coll_payoff = np.ones(len(coll_vec))-coll_vec
    return coll_payoff

def get_payoff_prog(curr_joint_state, next_joint_state, env, max_next_state):
    """
    returns: payoff array of shape (n, 1)
    """
    joint_progress = get_joint_agent_pos(next_joint_state, env.agent_progress_lines)-get_joint_agent_pos(curr_joint_state, env.agent_progress_lines)

    # normalizing bounds
    min_joint_progress = np.zeros(len(joint_progress))
    max_joint_progress = get_joint_agent_pos(max_next_state, env.agent_progress_lines)-get_joint_agent_pos(curr_joint_state, env.agent_progress_lines)
    
    #min_progress_0 = -np.max(Game.config.velocity_0)*Game.config.delta_t
    #min_progress_1 = -np.max(Game.config.velocity_1)*Game.config.delta_t

    # normalizing to [0,1]
    #min_progress_0 = -np.abs(max_agent_progress_0)
    #min_progress_1 = -np.abs(max_agent_progress_0)
    #max_progress_0 = np.abs(max_agent_progress_0)
    #max_progress_1 = np.abs(max_agent_progress_1)

    # calculate payoff
    denominator = max_joint_progress - min_joint_progress
    prog_payoff = np.where(denominator != 0, (joint_progress - min_joint_progress) / denominator, 0) # if denominator is zero, set payoff to zero

    return prog_payoff

def get_payoff_comp(curr_joint_state, next_joint_state, env, max_next_state, coll_dist):
    """
    calculating competitive payoff by taking measure if agent takes best possible action that maximizes its lead
    lead_payoff of one agent is the sum of the lead over all other agents
    returns: payoff array of shape (n, 1)
    """

    init_joint_pos = get_joint_env_pos(curr_joint_state, env.env_centerline)
    max_joint_pos = get_joint_env_pos(max_next_state, env.env_centerline)
    next_joint_pos = get_joint_env_pos(next_joint_state, env.env_centerline)

    # calculating competitive payoff for each agent by taking all other agents into account
    comp_payoff = np.zeros(len(curr_joint_state))
    for agent_a in range(len(init_joint_pos)):
        comp_payoff_a_sum = []
        for agent_b in range(len(init_joint_pos)):
            if agent_a == agent_b:
                continue
            # normalizing bounds
            extremum_lead_1 = max_joint_pos[agent_a]-max_joint_pos[agent_b]
            extremum_lead_2 = init_joint_pos[agent_a]-init_joint_pos[agent_a]
            extremum_lead_3 = init_joint_pos[agent_a]-max_joint_pos[agent_b]
            extremum_lead_4 = max_joint_pos[agent_a]-init_joint_pos[agent_b]

            min_lead = min(extremum_lead_1, extremum_lead_2, extremum_lead_3, extremum_lead_4)
            max_lead = max(extremum_lead_1, extremum_lead_2, extremum_lead_3, extremum_lead_4)

            lead_a = next_joint_pos[agent_a]-next_joint_pos[agent_b]

            if (max_lead - min_lead) > 0:
                lead_payoff_a = (lead_a - min_lead) / (max_lead - min_lead)
            else:
                lead_payoff_a = 0
            comp_payoff_a_sum.append(lead_payoff_a)
        comp_payoff[agent_a] = np.mean(comp_payoff_a_sum)
    
    return comp_payoff

    """init_env_pos_0 = get_env_progress(Game, prev_state.get_state(agent=0))
    init_env_pos_1 = get_env_progress(Game, prev_state.get_state(agent=1))

    next_env_pos_0 = get_env_progress(Game, next_state_obj.get_state(agent=0))
    next_env_pos_1 = get_env_progress(Game, next_state_obj.get_state(agent=1))

    full_env_pos_0 = get_env_progress(Game, max_next_state_0)
    full_env_pos_1 = get_env_progress(Game, max_next_state_1)

    extr_init_lead_0 = init_env_pos_0 - init_env_pos_1
    extr_full_lead_0 = full_env_pos_0 - full_env_pos_1
    extr_case1_lead_0 = init_env_pos_0 - full_env_pos_1
    extr_case2_lead_0 = full_env_pos_0 - init_env_pos_1

    extr_init_lead_1 = init_env_pos_1 - init_env_pos_0
    extr_full_lead_1 = full_env_pos_1 - full_env_pos_0
    extr_case1_lead_1 = init_env_pos_1 - full_env_pos_0
    extr_case2_lead_1 = full_env_pos_1 - init_env_pos_0

    min_lead_0 = min(extr_init_lead_0, extr_full_lead_0, extr_case1_lead_0, extr_case2_lead_0)
    min_lead_1 = min(extr_init_lead_1, extr_full_lead_1, extr_case1_lead_1, extr_case2_lead_1)
    max_lead_0 = max(extr_init_lead_0, extr_full_lead_0, extr_case1_lead_0, extr_case2_lead_0)
    max_lead_1 = max(extr_init_lead_1, extr_full_lead_1, extr_case1_lead_1, extr_case2_lead_1)

    lead_0 = next_env_pos_0 - next_env_pos_1
    lead_1 = next_env_pos_1 - next_env_pos_0

    weight_lead_0 = weight_lead
    weight_lead_1 = weight_lead

    if (max_lead_0 - min_lead_0) > 0:
        lead_payoff_0 = (lead_0 - min_lead_0) / (max_lead_0 - min_lead_0)
    else:
        lead_payoff_0 = 0
    if (max_lead_1 - min_lead_1) > 0:
        lead_payoff_1 = (lead_1 - min_lead_1) / (max_lead_1 - min_lead_1)
    else:
        lead_payoff_1 = 0"""

def get_payoff_time(final_joint_state):
    return np.zeros(len(final_joint_state))
    """if agent_has_finished(Game, final_state_obj, agent=0) and not agent_has_finished(Game, final_state_obj, agent=1):
        delta_time_0 = Game.config.max_timehorizon - final_state_obj.timestep
        delta_time_1 = 0
    elif not agent_has_finished(Game, final_state_obj, agent=0) and agent_has_finished(Game, final_state_obj, agent=1):
        delta_time_0 = 0
        delta_time_1 = Game.config.max_timehorizon - final_state_obj.timestep
    else:
        delta_time_0 = 0
        delta_time_1 = 0

    weight_time_0 = weight_timestep
    weight_time_1 = weight_timestep

    min_time_0 = 0
    min_time_1 = 0
    max_time_0 = Game.config.max_timehorizon - get_min_time_to_complete(Game)
    max_time_1 = Game.config.max_timehorizon - get_min_time_to_complete(Game)

    time_payoff_0 = (delta_time_0 - min_time_0) / (max_time_0 - min_time_0)
    time_payoff_1 = (delta_time_1 - min_time_1) / (max_time_1 - min_time_1)"""


def get_payoff_win(final_joint_state):
    return np.zeros(len(final_joint_state))
    """if agent_has_finished(Game, final_state_obj, agent=0) and not agent_has_finished(Game, final_state_obj, agent=1):
        winning_0 = 1
        winning_1 = 0
    elif not agent_has_finished(Game, final_state_obj, agent=0) and agent_has_finished(Game, final_state_obj, agent=1):
        winning_0 = 0
        winning_1 = 1
    else:
        winning_0 = 1
        winning_1 = 1

    weight_winning_0 = weight_winning
    weight_winning_1 = weight_winning

    min_winning_0 = 0
    min_winning_1 = 0
    max_winning_0 = 1
    max_winning_1 = 1

    winning_payoff_0 = (winning_0 - min_winning_0) / (max_winning_0 - min_winning_0)
    winning_payoff_1 = (winning_1 - min_winning_1) / (max_winning_1 - min_winning_1)"""

def get_payoff_lead(final_joint_state, env):
    # calculating competitive payoff for each agent by taking all other agents into account
    lead_payoff = np.zeros(len(final_joint_state))

    final_env_pos = get_joint_env_pos(final_joint_state, env.env_centerline)

    # normalizing bounds
    min_lead = -env.env_centerline['progress'][-1]
    max_lead = env.env_centerline['progress'][-1]

    for agent_a in range(len(final_joint_state)):
        comp_payoff_a_sum = []
        for agent_b in range(len(final_joint_state)):
            if agent_a == agent_b:
                continue

            lead_a = final_env_pos[agent_a]-final_env_pos[agent_b]
            lead_payoff_a = (lead_a - min_lead) / (max_lead - min_lead)
            comp_payoff_a_sum.append(lead_payoff_a)
        lead_payoff[agent_a] = np.mean(comp_payoff_a_sum)
    
    return lead_payoff

    progress_0 = get_env_progress(Game, final_state_obj.get_state(agent=0))
    progress_1 = get_env_progress(Game, final_state_obj.get_state(agent=1))
    lead_0 = progress_0 - progress_1
    lead_1 = progress_1 - progress_0

    weight_lead_0 = weight_final_lead
    weight_lead_1 = weight_final_lead

    min_lead_0 = -Game.env.env_centerline[-1][-1]
    min_lead_1 = -Game.env.env_centerline[-1][-1]
    max_lead_0 = Game.env.env_centerline[-1][-1]
    max_lead_1 = Game.env.env_centerline[-1][-1]

    lead_payoff_0 = (lead_0 - min_lead_0) / (max_lead_0 - min_lead_0)
    lead_payoff_1 = (lead_1 - min_lead_1) / (max_lead_1 - min_lead_1)



###################################
# Helper functions for get_payoff #
###################################

def interpolate_joint_states(start_state, end_state, n_interp=2):
    # interpolate between the start and end state
    interp_joint_states = np.zeros((n_interp, start_state.shape[0], start_state.shape[1]))
    for i in range(start_state.shape[0]):
        for j in range(start_state.shape[1]):
            interp_joint_states[:, i, j] = np.linspace(start_state[i, j], end_state[i, j], n_interp)
    return np.transpose(interp_joint_states, (1,2,0)) # transpose for convention

def search_collisions_interp(interp_states, coll_dist=0.5):
    """
    interp_states is (d, n, s) array with d interpolation depth, n agents, and s states
    """
    n_agents = interp_states.shape[1]
    # the following vector will store a 1 if there is a collision and 0 if there is no collision
    coll_vec = np.zeros(n_agents)
    for joint_state in interp_states:
        coll_vec_update = coll_count(joint_state, coll_dist)
        coll_vec = np.maximum(coll_vec, coll_vec_update)
    return coll_vec

        



"""

def get_intermediate_payoffs(curr_timestep, curr_joint_state, next_joint_state, config_agent, config_global, discount_factor=0):


def calculate_lead_payoff(Game, prev_state, next_state_obj, max_next_state_0, max_next_state_1, weight_lead):
    

    weights_0['weight_lead_0'] = weight_lead_0
    payoffs_0['lead_payoff_0'] = lead_payoff_0
    weights_1['weight_lead_1'] = weight_lead_1
    payoffs_1['lead_payoff_1'] = lead_payoff_1

def calculate_time_payoff(Game, final_state_obj, weight_timestep):
    

    weights_0['weight_time_0'] = weight_time_0
    weights_1['weight_time_1'] = weight_time_1
    payoffs_0['final_time_payoff_0'] = time_payoff_0
    payoffs_1['final_time_payoff_1'] = time_payoff_1

def calculate_lead_of_agents(Game, final_state_obj, weight_final_lead):
    

    weights_0['weight_lead_0'] = weight_lead_0
    weights_1['weight_lead_1'] = weight_lead_1
    payoffs_0['final_lead_payoff_0'] = lead_payoff_0
    payoffs_1['final_lead_payoff_1'] = lead_payoff_1

def calculate_agent_winning(Game, final_state_obj, weight_winning):
    

    weights_0['weight_winning_0'] = weight_winning_0
    weights_1['weight_winning_1'] = weight_winning_1
    payoffs_0['final_winning_payoff_0'] = winning_payoff_0
    payoffs_1['final_winning_payoff_1'] = winning_payoff_1


def calculate_total_payoffs(Game, interm_payoff_list=None, final_payoff_list=None, weight_interm, weight_final):
    if len(interm_payoff_list) == 0:
        interm_payoff_list = [[0] * len(Game.Model_params['agents'])]
    if len(final_payoff_list) == 0:
        final_payoff_list = [[0] * len(Game.Model_params['agents'])]

    min_total_interm = 0
    max_total_interm = np.sum([1 / ((1 + Game.config.discount_factor) ** k) for k in range(len(interm_payoff_list))])

    min_total_final = 0
    max_total_final = 1

    total_payoff_list = []
    for agent in Game.Model_params['agents']:
        interm_sum = sum(interm[agent] for interm in interm_payoff_list)
        total_payoff_interm = (interm_sum - min_total_interm) / (max_total_interm - min_total_interm)

        final_sum = sum(final[agent] for final in final_payoff_list)
        total_payoff_final = (final_sum - min_total_final) / (max_total_final - min_total_final)

        epsilon = 1e-10
        payoffs_weighted = (weight_interm * total_payoff_interm) / (weight_interm + weight_final + epsilon) + (
                    weight_final * total_payoff_final) / (weight_interm + weight_final + epsilon)

        total_payoff_list.append(payoffs_weighted)

    return total_payoff_list

def get_final_payoffs(Game, final_state_obj, discount_factor=0):
    discount = 1 / ((1 + discount_factor) ** final_state_obj.timestep)

    weights_0 = {}
    weights_1 = {}
    payoffs_0 = {}
    payoffs_1 = {}

    calculate_time_payoff(Game, final_state_obj, Game.config.weight_timestep)
    calculate_lead_of_agents(Game, final_state_obj, Game.config.weight_final_lead)
    calculate_agent_winning(Game, final_state_obj, Game.config.weight_winning)

    total_weight_0 = sum(weights_0.values()) + epsilon
    total_weight_1 = sum(weights_1.values()) + epsilon

    payoffs_0 = {key: value * discount for key, value in payoffs_0.items()}
    payoffs_1 = {key: value * discount for key, value in payoffs_1.items()}

    weighted_payoff_0 = sum(w * p for w, p in zip(weights_0.values(), payoffs_0.values())) / total_weight_0
    weighted_payoff_1 = sum(w * p for w, p in zip(weights_1.values(), payoffs_1.values())) / total_weight_1

    final_payoffs_sum = [weighted_payoff_0, weighted_payoff_1]
    final_payoffs_each = {**payoffs_0, **payoffs_1}

    return final_payoffs_sum, final_payoffs_each

def get_total_payoffs(Game, interm_payoff_list=None, final_payoff_list=None):
    if len(interm_payoff_list) == 0:
        interm_payoff_list = [[0] * len(Game.Model_params['agents'])]
    if len(final_payoff_list) == 0:
        final_payoff_list = [[0] * len(Game.Model_params['agents'])]

    total_payoff_list = calculate_total_payoffs(Game, interm_payoff_list, final_payoff_list, Game.config.weight_interm, Game.config.weight_final)

    return total_payoff_list
"""