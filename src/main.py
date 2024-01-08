import io
import sys
from mcts import State, MCTSNode
from common import *
from environment import *
from kinodynamic import *
from payoff_utilities import *
from csv_utilities import *
from networkx_utilities import *

def MCTS_play_subgame(simhorizon, node_current_timestep, max_payoff, min_payoff, payoff_range, interm_weights_vec, final_weights_vec, forbidden_states):
    #num_iter = int(1-node_current_timestep.state.timestep/timehorizon)*num_iter

    """interm_payoff_vec = np.zeros((Model_params["len_interm_payoffs"],1))
    final_payoff_vec = np.zeros((Model_params["len_final_payoffs"],1))"""
    
    for iter in range(MCTS_params['num_iter']):
        print("Horizon {} | Iteration {}".format(simhorizon, iter))
        #print("Starting tree policy")
        v = node_current_timestep._tree_policy(payoff_range, forbidden_states)

        #print("Starting rollout")
        rollout_trajectory, interm_payoff_rollout, final_payoff_rollout, forbidden_states = v.rollout(interm_weights_vec, final_weights_vec, forbidden_states)
        payoff_list = get_total_payoffs_all_agents(interm_payoff_rollout, final_payoff_rollout)
        max_payoff, min_payoff, payoff_range = update_payoff_range(max_payoff, min_payoff, payoff_list)
        

        # write every x rollout trajectories
        if iter % freq_stat_data == 0:
            csv_write_rollout_last(rollout_trajectory, timehorizon = node_current_timestep.state.timestep)
        
        #print("Backpropagating")
        v.backpropagate(payoff_list)
        #payoff_weights = update_weigths_payoff(node_current_timestep, payoff_weights)
        #print("Payoff weights: {}".format(payoff_weights))
    
    #selected_action = node_current_timestep.select_action(payoff_range)
    
    #next_node = node_current_timestep.select_child(payoff_range, forbidden_states)
    next_node = node_current_timestep.robust_child()

    save_tree_to_file(node_current_timestep, path_to_tree.format(node_current_timestep.state.timestep))
    return next_node.state, next_node.parent_action, forbidden_states

if __name__ == "__main__":
    # create a text trap and redirect stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap

    # initialize csv files
    csv_init_global_state()
    init_tree_file()   

    # forbidden states
    forbidden_states = [] # list of states [x, y, theta, timestep]

    # initialize root node
    intial_state = State(x0=env.init_state['x0'], y0=env.init_state['y0'], theta0=env.init_state['theta0'],
                         x1=env.init_state['x1'], y1=env.init_state['y1'], theta1=env.init_state['theta1'],
                         timestep=env.init_state['timestep'])
    root_node = MCTSNode(intial_state, forbidden_states=forbidden_states)
    node_current_horizon = root_node

    # initialize trajectory
    trajectory = [root_node.state.get_state_together()]
    csv_write_global_state(root_node.state)

    # initialize payoff list
    interm_payoff_vec_global = np.zeros((Model_params["len_interm_payoffs"],1))
    final_payoff_vec_global = np.zeros((Model_params["len_final_payoffs"],1))
    payoff_dict_global = {}
    for agent in Model_params["agents"]:
        payoff_dict_global[agent] = [(0,0)]

    

    while not is_terminal(node_current_horizon.state):
        csv_init_rollout_last()
        
        simhorizon = node_current_horizon.state.timestep

        interm_weights_vec = init_interm_weights(node_current_horizon)
        final_weights_vec = init_final_weights(node_current_horizon)
        
        

        next_state, chosen_action, forbidden_states = MCTS_play_subgame(simhorizon, node_current_horizon, max_payoff, min_payoff, payoff_range, interm_weights_vec, final_weights_vec, forbidden_states)

        # update payoffs global solution
        interm_payoff_vec_global = update_intermediate_payoffs(node_current_horizon.state, next_state, interm_payoff_vec_global, interm_weights_vec)
        payoff_list = get_total_payoffs_all_agents(interm_payoff_vec_global, final_payoff_vec_global)
        for agent in Model_params["agents"]:
            payoff_dict_global[agent].append((simhorizon+1, payoff_list[agent]))

        trajectory.append(next_state.get_state_together())

        csv_write_global_state(next_state)
        node_current_horizon = MCTSNode(next_state, forbidden_states=forbidden_states) 

    # update final payoffs global solution
    final_payoff_vec_global = update_final_payoffs(node_current_horizon.state, final_payoff_vec_global, final_weights_vec)
    payoff_list = get_total_payoffs_all_agents(interm_payoff_vec_global, final_payoff_vec_global)
    for agent in Model_params["agents"]:
        payoff_dict_global[agent].append((simhorizon+1, payoff_list[agent]))

# save final payoffs to textfile
    with open(os.path.join(path_to_results, next_video_name + ".txt"), 'a') as f:
        f.write(f"Environment trigger: {env.env_name_trigger}\n")
        for agent in Model_params["agents"]:
            f.write(f"Payoff Agent {agent}: {payoff_dict_global[agent]}\n")