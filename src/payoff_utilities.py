from environment import *
from common import *
from kinodynamic import *


def get_intermediate_penalty(next_state):

    dist_agents = distance(next_state.get_state_0()[:2], next_state.get_state_1()[:2])

    # exponential penalty for collision
    collision_0 = -Model_params["payoff_vector"]["intermediate_penalties"]["penalty_collision_0"]["weight"]*np.exp(-dist_agents)
    collision_1 = -Model_params["payoff_vector"]["intermediate_penalties"]["penalty_collision_0"]["weight"]*np.exp(-dist_agents)

    # payoff at every timestep
    penalty_0 = collision_0
    penalty_1 = collision_1

    return penalty_0, penalty_1

def get_intermediate_reward(prev_state_obj, next_state_obj):
    advancement_0 = next_state_obj.get_state_0()[0]-prev_state_obj.get_state_0()[0]
    advancement_1 = next_state_obj.get_state_1()[0]-prev_state_obj.get_state_1()[0]
    # reward for making progress
    progress_0 = Model_params["payoff_vector"]["intermediate_rewards"]["reward_progress_0"]["weight"]*advancement_0
    progress_1 = Model_params["payoff_vector"]["intermediate_rewards"]["reward_progress_1"]["weight"]*advancement_1

    # payoff at every timestep
    reward_0 = progress_0
    reward_1 = progress_1

    return reward_0, reward_1

def get_final_payoffs(final_state):
    # final utility at final timestep k=T
    lead_0 = Model_params["payoff_vector"]["final_rewards"]["reward_lead_0"]["weight"]*(final_state.x0 - final_state.x1)
    lead_1 = Model_params["payoff_vector"]["final_rewards"]["reward_lead_0"]["weight"]*(final_state.x1 - final_state.x0)

    # payoff at final timestep
    payoff_0 = lead_0
    payoff_1 = lead_1

    return payoff_0, payoff_1

def update_payoff_range(max_payoff, min_payoff, payoff_vector):
    payoff_list = [0] * len(Model_params["agents"])
    for payoffs in Model_params["payoff_vector"].values():
        for payoff in payoffs.values():
            for agent in Model_params["agents"]:
                if payoff["agent"] == agent:
                    payoff_list[agent] += float(payoff_vector[payoff["pos"]])
    
    biggest_payoff = max(payoff_list)
    if biggest_payoff > max_payoff:
        max_payoff = biggest_payoff
    
    smallest_payoff = min(payoff_list)
    if smallest_payoff < min_payoff:
        min_payoff = smallest_payoff
    
    payoff_range = max_payoff - min_payoff
    return max_payoff, min_payoff, payoff_range


def init_payoff_weights(MCTS_node):
    weigths_payoff_vector = np.zeros((Model_params["len_payoff_vector"],1))
    for component in Model_params["payoff_vector"].values():
        for payoff_agent in component.values():
            weigths_payoff_vector[payoff_agent["pos"]] = payoff_agent["weight"]
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