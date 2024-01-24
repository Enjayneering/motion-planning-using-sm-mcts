from common import *
from kinodynamic_utilities import *

from config import *

def get_intermediate_payoffs(Game, prev_state_obj, next_state_obj):
    #print("prev_state_obj: {}".format(prev_state_obj.get_state_together()))
    #print("next_state_obj: {}".format(next_state_obj.get_state_together()))
    dist_agents = distance(next_state_obj.get_state_0(), next_state_obj.get_state_1())
    progress_0 = (next_state_obj.get_state_0()[0]-prev_state_obj.get_state_0()[0])
    progress_1 = (next_state_obj.get_state_1()[0]-prev_state_obj.get_state_1()[0])
    discount = Game.config.discount_factor**prev_state_obj.timestep

    update_vec = discount*np.array([[np.exp(-dist_agents)], [np.exp(-dist_agents)], [progress_0], [progress_1]])
    interm_payoff_vec = Game.interm_weights_vec*update_vec
    return interm_payoff_vec

def get_final_payoffs(Game, final_state):
    lead_0 = final_state.x0 - final_state.x1
    lead_1 = final_state.x1 - final_state.x0
    time_0 = final_state.timestep
    time_1 = final_state.timestep

    update_vec = np.array([[time_0], [time_1], [lead_0], [lead_1]])
    final_payoff_vec = Game.final_weights_vec*update_vec
    return final_payoff_vec

def get_total_payoffs_all_agents(Game, interm_payoff_vec=None, final_payoff_vec=None):
    payoff_list = [0] * len(Game.Model_params["agents"])
    for agent in Game.Model_params["agents"]:
        interm_payoffs = 0
        final_payoffs = 0

        if interm_payoff_vec is not None:
            for payoff in Game.Model_params["interm_payoffs"].keys():
                if Game.Model_params["interm_payoffs"][payoff]["agent"] == agent:
                    interm_payoffs += interm_payoff_vec[Game.Model_params["interm_payoffs"][payoff]["pos"]]

        if final_payoff_vec is not None:
            for payoff in Game.Model_params["final_payoffs"].keys():
                if Game.Model_params["final_payoffs"][payoff]["agent"] == agent:
                    final_payoffs += final_payoff_vec[Game.Model_params["final_payoffs"][payoff]["pos"]]
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

def init_interm_weights(Game):
    weigths_payoff_vector = np.zeros((Game.Model_params["len_interm_payoffs"],1))
    for interm_payoff in Game.Model_params["interm_payoffs"].values():
        weigths_payoff_vector[interm_payoff["pos"]] = interm_payoff["weight"]
    return weigths_payoff_vector

def init_final_weights(Game):
    weigths_payoff_vector = np.zeros((Game.Model_params["len_final_payoffs"],1))
    for final_payoff in Game.Model_params["final_payoffs"].values():
        weigths_payoff_vector[final_payoff["pos"]] = final_payoff["weight"]
    return weigths_payoff_vector


"""def update_weigths_payoff(MCTS_node, payoff_weights):
    aver_payoff_vector = MCTS_node.aggr_payoff_vector / MCTS_node._number_of_visits

    # calculate update_vector
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
    updated_payoff_weights = update_vector*payoff_weights
    risk_factor_0 = 0.5
    risk_factor_1 = 0.5
    new_weights = np.array([risk_factor_0*(np.abs(aver_payoff_vector[2])/np.abs(aver_payoff_vector[0])), risk_factor_1*(np.abs(aver_payoff_vector[3])/np.abs(aver_payoff_vector[1])), 1, 1]).T
    return new_weights"""

"""
from competitive_game import CompetitiveGame
from mcts_utilities import State, MCTSNode
if __name__ == "__main__":
    dragrace_config = Config(
            env_name_trigger = [(0,'benchmark_static_dragrace')],
            num_sim=1,
            num_iter=1000,
            max_timehorizon=3,
            c_param = np.sqrt(2),
            penalty_distance_0=-1,
            penalty_distance_1=-1,
            reward_progress_0=1,
            reward_progress_1=1,
            penalty_timestep_0=-1,
            penalty_timestep_1=-1,
            reward_lead_0=10,
            reward_lead_1=10,
            velocity_0=np.linspace(0, 2, 3).tolist(),
            ang_velocity_0=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            velocity_1=np.linspace(0, 1, 2).tolist(),
            ang_velocity_1=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            )
    Game = CompetitiveGame(exp_config)
    
    # Optimal trajectories [self.x0, self.y0, self.theta0, self.x1, self.y1, self.theta1, self.timestep]
    traj_drag_race = [[1, 1, 0, 1, 2, 0, 0],
                      [3, 1, 0, 2, 2, 0, 1],
                      [5, 1, 0, 3, 2, 0, 2]]
    payoff_vec = np.zeros((Game.Model_params["len_interm_payoffs"],1))
    interm_payoff_vec = np.zeros((Game.Model_params["len_interm_payoffs"],1))
    final_payoff_vec = np.zeros((Game.Model_params["len_final_payoffs"],1))
    state_now = State(traj_drag_race[0])
    for state in traj_drag_race[1:]:
        state_next = State(state)
        interm_payoff_vec = update_intermediate_payoffs(Game, interm_payoff_vec, state_now, state_next)"""
        
