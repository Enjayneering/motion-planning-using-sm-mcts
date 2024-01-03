from environment import *
from common import *
from kinodynamic import *


def get_intermediate_penalty(next_state, payoff_weights):

    # intermediate reward at timestep k
    dist_agents = distance(next_state.get_state_0()[:2], next_state.get_state_1()[:2])

    collision_0 = -payoff_weights[0]*np.exp(-dist_agents*Model_params["payoff_vector"]["intermediate_penalties"]["penalty_collision_0"]["weight"]) if dist_agents < 1 else 0
    collision_1 = -payoff_weights[1]*np.exp(-dist_agents*Model_params["payoff_vector"]["intermediate_penalties"]["penalty_collision_1"]["weight"]) if dist_agents < 1 else 0

    # payoff at every timestep
    penalty_0 = collision_0
    penalty_1 = collision_1

    return penalty_0, penalty_1

def get_final_payoffs(final_state):
    # final utility at final timestep k=T
    lead_0 = Model_params["payoff_vector"]["final_rewards"]["reward_lead_0"]["weight"]*(final_state.x0 - final_state.x1)
    lead_1 = Model_params["payoff_vector"]["final_rewards"]["reward_lead_0"]["weight"]*(final_state.x1 - final_state.x0)

    # payoff at final timestep
    payoff_0 = lead_0
    payoff_1 = lead_1

    return payoff_0, payoff_1

def update_payoff_range(max_payoff, min_payoff, payoff_vector):
    payoff_0 = payoff_vector[0]+payoff_vector[2]
    payoff_1 = payoff_vector[1]+payoff_vector[3]
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