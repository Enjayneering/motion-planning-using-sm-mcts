from environment import *
from common import *
from kinodynamic import *


def update_intermediate_payoffs(prev_state_obj, next_state_obj, interm_payoff_vec, interm_weights_vec):
    dist_agents = distance(next_state_obj.get_state_0()[:2], next_state_obj.get_state_1()[:2])
    advancement_0 = (next_state_obj.get_state_0()[0]-prev_state_obj.get_state_0()[0])/(next_state_obj.timestep/env.max_timehorizon)
    advancement_1 = (next_state_obj.get_state_1()[0]-prev_state_obj.get_state_1()[0])/(next_state_obj.timestep/env.max_timehorizon)

    update_vec = np.array([[np.exp(-dist_agents)], [np.exp(-dist_agents)], [advancement_0], [advancement_1]])
    interm_payoff_vec += interm_weights_vec*update_vec
    return interm_payoff_vec

def update_final_payoffs(final_state, final_payoff_vec, final_weights_vec):
    lead_0 = final_state.x0 - final_state.x1
    lead_1 = final_state.x1 - final_state.x0
    time_0 = final_state.timestep
    time_1 = final_state.timestep

    update_vec = np.array([[time_0], [time_1], [lead_0], [lead_1]])
    final_payoff_vec += final_weights_vec*update_vec
    return final_payoff_vec

def get_total_payoffs_all_agents(interm_payoff_vec, final_payoff_vec):
    payoff_list = [0] * len(Model_params["agents"])
    for agent in Model_params["agents"]:
        interm_payoffs = 0
        final_payoffs = 0
        for interm_payoff in Model_params["interm_payoffs"].values():
            if interm_payoff["agent"] == agent:
                interm_payoffs += interm_payoff_vec[interm_payoff["pos"]]
        for final_payoff in Model_params["final_payoffs"].values():
            if final_payoff["agent"] == agent:
                final_payoffs += final_payoff_vec[final_payoff["pos"]]
        total_payoff = interm_payoffs+final_payoffs
        payoff_list[agent] = float(total_payoff)
    return payoff_list

def update_payoff_range(max_payoff, min_payoff, payoff_list):
    """payoff_list = [0] * len(Model_params["agents"])
    for payoffs in Model_params["payoff_vector"].values():
        for payoff in payoffs.values():
            for agent in Model_params["agents"]:
                if payoff["agent"] == agent:
                    payoff_list[agent] += float(payoff_vector[payoff["pos"]])"""
    biggest_payoff = max(payoff_list)
    if biggest_payoff > max_payoff:
        max_payoff = biggest_payoff
    
    smallest_payoff = min(payoff_list)
    if smallest_payoff < min_payoff:
        min_payoff = smallest_payoff
    
    payoff_range = max_payoff - min_payoff

    return max_payoff, min_payoff, payoff_range


def init_interm_weights(MCTS_node):
    weigths_payoff_vector = np.zeros((Model_params["len_interm_payoffs"],1))
    for interm_payoff in Model_params["interm_payoffs"].values():
        weigths_payoff_vector[interm_payoff["pos"]] = interm_payoff["weight"]
    return weigths_payoff_vector

def init_final_weights(MCTS_node):
    weigths_payoff_vector = np.zeros((Model_params["len_final_payoffs"],1))
    for final_payoff in Model_params["final_payoffs"].values():
        weigths_payoff_vector[final_payoff["pos"]] = final_payoff["weight"]
    return weigths_payoff_vector


def update_weigths_payoff(MCTS_node, payoff_weights):
    aver_payoff_vector = MCTS_node.aggr_payoff_vector / MCTS_node._number_of_visits

    """# calculate update_vector
    A = np.eye(len(payoff_weights))

    # fill A with respective final reward values
    for reward in Model_params["payoff_vector"]["final_rewards"].values():
        agent = reward["agent"]
        pos = reward["pos"]
        final_reward_factor = np.abs(aver_payoff_vector[pos])
        A[pos, pos] = final_reward_factor
        for penalty in Model_params['payoff_vector']['intermediate_penalties'].values():
            if penalty['agent'] == agent:
                pos = penalty['pos']
                A[pos, pos] = final_reward_factor

    b = np.abs(np.reciprocal(aver_payoff_vector))
    update_vector = A@b
    updated_payoff_weights = update_vector*payoff_weights"""
    risk_factor_0 = 0.5
    risk_factor_1 = 0.5
    new_weights = np.array([risk_factor_0*(np.abs(aver_payoff_vector[2])/np.abs(aver_payoff_vector[0])), risk_factor_1*(np.abs(aver_payoff_vector[3])/np.abs(aver_payoff_vector[1])), 1, 1]).T
    return new_weights