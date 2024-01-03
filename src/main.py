import io
import sys
from mcts import State, MCTSNode
from common import *
from environment import *
from kinodynamic import *
from payoff_utilities import *
from csv_utilities import *
from networkx_utilities import save_tree_to_file

def MCTS_play_subgame(node_current_timestep, max_payoff, min_payoff, payoff_range):
    #num_iter = int(1-node_current_timestep.state.timestep/timehorizon)*num_iter
    
    payoff_weights = init_payoff_weights(node_current_timestep)

    for iter in range(MCTS_params['num_iter']):
        print("Starting tree policy")
        v = node_current_timestep._tree_policy(payoff_range)

        print("Starting rollout")
        rollout_trajectory, payoff_vector = v.rollout(payoff_weights)
        max_payoff, min_payoff, payoff_range = update_payoff_range(max_payoff, min_payoff, payoff_vector)
        

        # write every x rollout trajectories
        if iter % freq_stat_data == 0:
            csv_write_rollout_last(rollout_trajectory, timehorizon = node_current_timestep.state.timestep)
        
        print("Backpropagating")
        v.backpropagate(payoff_vector)
        payoff_weights = update_weigths_payoff(node_current_timestep, payoff_weights)
        print("Payoff weights: {}".format(payoff_weights))
        #node_current_timestep.update_action_stats()
    #selected_action = node_current_timestep.select_action(payoff_range)
    #next_state = node_current_timestep.select_child(payoff_range).state

    next_node = node_current_timestep.robust_child()

    save_tree_to_file(node_current_timestep, path_to_tree.format(node_current_timestep.state.timestep))
    return next_node.state, next_node.parent_action

if __name__ == "__main__":
    # create a text trap and redirect stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap

    # initialize csv files
    csv_init_global_state()

    # initialize root node
    intial_state = State(x0=env.init_state['x0'], y0=env.init_state['y0'], theta0=env.init_state['theta0'],
                         x1=env.init_state['x1'], y1=env.init_state['y1'], theta1=env.init_state['theta1'],
                         timestep=env.init_state['timestep'])
    root_node = MCTSNode(intial_state)
    node_current_horizon = root_node

    # initialize trajectory
    trajectory = [root_node.state.get_state_together()]
    csv_write_global_state(root_node.state)

    # initialize payoff vector
    payoff_0 = 0
    payoff_1 = 0
    payoff_list_0 = []
    payoff_list_1 = []

    while not is_terminal(node_current_horizon.state):
        simhorizon = node_current_horizon.state.timestep
        print("\nSimulation horizon: {}\n".format(simhorizon))
        csv_init_rollout_last()

        next_state, chosen_action = MCTS_play_subgame(node_current_horizon, max_payoff, min_payoff, payoff_range)

        # update payoffs global solution
        payoff_0_new, payoff_1_new = get_intermediate_penalty(next_state, MCTS_params['penalty_collision_init'])
        payoff_0 += payoff_0_new
        payoff_list_0.append((simhorizon, payoff_0))
        payoff_1 += payoff_1_new
        payoff_list_1.append((simhorizon, payoff_1))

        trajectory.append(next_state.get_state_together())

        csv_write_global_state(next_state)
        node_current_horizon = MCTSNode(next_state)

    # update final payoffs global solution
    payoff_0_new, payoff_1_new = get_final_payoffs(node_current_horizon.state)
    payoff_0 += payoff_0_new
    payoff_list_0.append((simhorizon, payoff_0))
    payoff_1 += payoff_1_new
    payoff_list_1.append((simhorizon, payoff_1))

    # save final payoffs to textfile
    with open(os.path.join(path_to_results, next_video_name + ".txt"), 'a') as f:
        f.write(f"Environment trigger: {env.env_name_trigger}\n")
        f.write(f"Payoff Agent 0: {payoff_list_0}\n")
        f.write(f"Payoff Agent 1: {payoff_list_1}\n")