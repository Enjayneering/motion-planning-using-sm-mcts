import io
import os
import sys
import time

from config import *
from mcts_utilities import State, MCTSNode
from common import *
from environment_utilities import *
from kinodynamic_utilities import *
from payoff_utilities import *
from csv_utilities import *
from networkx_utilities import *


class CompetitiveGame:
    def __init__(self, config):
        self.config = config
        self.env = Environment(self.config)
        self.name = get_next_game_name(path_to_results, self)

        self.MCTS_params = self._set_mcts_params()
        self.Model_params = self._set_model_params()
        self.Kinodynamic_params = self._set_Kinodynamic_params()

        self.forbidden_states = []
        self.mcts_nodes_global = [self._init_root_node()] # stores all nodes of the tree, represents the final trajectory

        # initialize parameters and weights
        self.max_payoff = 0
        self.min_payoff = 0
        self.payoff_range = 0
        self.interm_weights_vec = init_interm_weights(self)
        self.final_weights_vec = init_final_weights(self)
        
        self.interm_payoff_vec_global = np.zeros((self.Model_params["len_interm_payoffs"],1))
        self.final_payoff_vec_global = np.zeros((self.Model_params["len_final_payoffs"],1))
        self.payoff_dict_global = self._init_payoff_dict_global() # stores all payoff data for each timestep of the horizon

    def _set_model_params(self):
        Model_params = {
            "state_space": ['x0', 'y0', 'theta0', 'x1', 'y1', 'theta1', 'timestep'],
            "delta_t": self.config.delta_t,
            "agents": [0, 1],
            "interm_payoffs": {
                "penalty_distance_0": {"pos": 0, "weight": self.config.penalty_distance_0, "agent": 0},
                "penalty_distance_1": {"pos": 1, "weight": self.config.penalty_distance_1, "agent": 1},
                "reward_progress_0": {"pos": 2, "weight": self.config.reward_progress_0, "agent": 0},
                "reward_progress_1": {"pos": 3, "weight": self.config.reward_progress_1, "agent": 1},
                "penalty_agressor_0": {"pos": 4, "weight": self.config.penalty_agressor_0, "agent": 0},
                "penalty_agressor_1": {"pos": 5, "weight": self.config.penalty_agressor_1, "agent": 1},
            },
            "final_payoffs": {
                "penalty_timestep_0": {"pos": 0, "weight": self.config.penalty_timestep_0, "agent": 0},
                "penalty_timestep_1": {"pos": 1, "weight": self.config.penalty_timestep_1, "agent": 1},
                "reward_lead_0": {"pos": 2, "weight": self.config.reward_lead_0, "agent": 0},
                "reward_lead_1": {"pos": 3, "weight": self.config.reward_lead_1, "agent": 1},
            },
            }
        Model_params["len_interm_payoffs"] = len(Model_params["interm_payoffs"])
        Model_params["len_final_payoffs"] = len(Model_params["final_payoffs"])
        return Model_params
    
    def _set_Kinodynamic_params(self):
        Kinodynamic_params = {
        "action_set_0": {"velocity_0": self.config.velocity_0,
                        "ang_velocity_0": self.config.ang_velocity_0},
        "action_set_1": {"velocity_1": self.config.velocity_1,
                        "ang_velocity_1": self.config.ang_velocity_1},
        }
        return Kinodynamic_params

    def _set_mcts_params(self):
        MCTS_params = {
        "num_iter": self.config.num_iter, #max number of simulations, proportional to complexity of the game
        "c_param": self.config.c_param, # c_param: exploration parameter | 3.52 - Tuned from Paper by Perick, 2012

        #'penalty_collision_init': 0.1, # penalty at initial state
        #'penalty_collision_delay': 1, # dynamic factor for ensuring reward is bigger than sum of penalties
        #"penalty_stuck_in_env": -1,
        }
        return MCTS_params
        
    def _init_root_node(self):
        # initialize root node
        initial_state = State(x0=self.env.init_state['x0'], 
                            y0=self.env.init_state['y0'], 
                            theta0=self.env.init_state['theta0'],
                            x1=self.env.init_state['x1'], 
                            y1=self.env.init_state['y1'], 
                            theta1=self.env.init_state['theta1'],
                            timestep=self.env.init_state['timestep'])
        root_node = MCTSNode(self, initial_state)
        return root_node
    
    def _init_payoff_dict_global(self):
        # initialize payoff list
        payoff_dict_global = {}
        for payoff in self.Model_params["interm_payoffs"].keys():
            payoff_dict_global[payoff]= []
        for payoff in self.Model_params["final_payoffs"].keys():
            payoff_dict_global[payoff]= []
        for agent in self.Model_params["agents"]:
            payoff_dict_global["payoff_total"+str(agent)] = []
        return payoff_dict_global
    
    """def _update_payoff_dict(self, next_payoff_dict):
        for agent in self.Model_params["agents"]:
            for interm_payoff in self.Model_params["interm_payoffs"].values():
                self.payoff_dict_global[agent].append(next_payoff_dict[agent][interm_payoff])
            for final_payoff in self.Model_params["final_payoffs"].values():
                self.payoff_dict_global[agent].append(next_payoff_dict[agent][final_payoff])
        return self.payoff_dict_global"""
    
    def search_game_tree(self, current_node):
        current_node_mcts = current_node
        simhorizon = current_node_mcts.state.timestep
        
        # TREE POLICY
        for iter in range(self.MCTS_params['num_iter']):
            #print("Horizon {} | Iteration {}".format(simhorizon, iter))
            v = current_node_mcts._tree_policy(self)

            #print("Starting rollout")
            rollout_trajectory, interm_payoff_rollout_vec, final_payoff_rollout_vec = v.rollout(self)
            
            payoff_total_list = get_total_payoffs_all_agents(self, interm_payoff_rollout_vec, final_payoff_rollout_vec)
            self.max_payoff, self.min_payoff, self.payoff_range = update_payoff_range(self.max_payoff, self.min_payoff, payoff_total_list)

            # write every x rollout trajectories
            if iter % freq_stat_data == 0:
                csv_write_rollout_last(self, rollout_trajectory, timehorizon = current_node_mcts.state.timestep, config=self.config)
            
            #print("Backpropagating")
            v.backpropagate(self, payoff_total_list)

            if is_feature_active(self.config.feature_flags["payoff_weights"]["adaptive"]):
                #self.interm_weights_vec = update_interm_weights(self, self.interm_weights_vec, interm_payoff_rollout_vec)
                #self.final_weights_vec = update_final_weights(self, self.final_weights_vec, final_payoff_rollout_vec)
                #payoff_weights = update_weigths_payoff(current_node_mcts, payoff_weights)
                pass
        
        if is_feature_active(self.config.feature_flags["final_move"]["robust"]):
            next_node = current_node_mcts.robust_child()
        elif is_feature_active(self.config.feature_flags["final_move"]["max"]):
            pass

        if not experimental_mode:
            save_tree_to_file(current_node_mcts, path_to_tree.format(current_node_mcts.state.timestep))
            #print("Next State: {}".format(next_node.state.get_state_together()))
        return next_node
    
    def compute_trajectories(self):
        # create a text trap and redirect stdout
        #text_trap = io.StringIO()
        #sys.stdout = text_trap

        current_node = self.mcts_nodes_global[0]

        if not experimental_mode:
            csv_init_global_state(self)
            init_tree_file()   
            csv_write_global_state(self, current_node.state)
            test_write_params(self)
        elif experimental_mode:
            result_dict = {}

        # RUN TRAJECTORY PLANNER
        start_time = time.time()

        while not is_terminal(self, current_node.state):
            print("Searching game tree in timestep {}...".format(current_node.state.timestep))
            if not experimental_mode:
                csv_init_rollout_last(self)
            elif experimental_mode:
                pass
                
            if is_feature_active(self.config.feature_flags["payoff_weights"]["adaptive"]):
                self.interm_weights_vec = init_interm_weights(self)
                self.final_weights_vec = init_final_weights(self)
            elif is_feature_active(self.config.feature_flags["payoff_weights"]["fixed"]):
                self.interm_weights_vec = init_interm_weights(self)
                self.final_weights_vec = init_final_weights(self)
            
            # RUN SINGLE MCTS ALGORITHM IN CURRENT TIMESTEP
            next_node = self.search_game_tree(current_node)

            interm_payoff_increment_vec = get_intermediate_payoffs(self, current_node.state, next_node.state)
            self.interm_payoff_vec_global += interm_payoff_increment_vec

            # update global payoff dict
            for payoff in self.Model_params["interm_payoffs"].keys():
                self.payoff_dict_global[payoff].append(float(interm_payoff_increment_vec[self.Model_params["interm_payoffs"][payoff]["pos"]]))
            for agent in self.Model_params["agents"]:
                total_payoff = get_total_payoffs_all_agents(self, self.interm_payoff_vec_global, self.final_payoff_vec_global)[agent]
                self.payoff_dict_global["payoff_total"+str(agent)].append(total_payoff)

            if not experimental_mode:
                csv_write_global_state(self, self.mcts_nodes_global[-1].state)
            
            self.mcts_nodes_global.append(next_node)
            current_node = next_node
        
        if not experimental_mode:
                csv_write_global_state(self, self.mcts_nodes_global[-1].state)
        print("Terminal state: {}".format(current_node.state.get_state_together()))
        
        # update final payoffs global solution
        final_payoff_increment_vec = get_final_payoffs(self, current_node.state)    
        self.final_payoff_vec_global += final_payoff_increment_vec

        end_time = time.time()
        print("Runtime: {} s".format(end_time - start_time))

        # update global payoff dict
        for payoff in self.Model_params["final_payoffs"].keys():
            self.payoff_dict_global[payoff].append(float(final_payoff_increment_vec[self.Model_params["final_payoffs"][payoff]["pos"]]))
        for agent in self.Model_params["agents"]:
            total_payoff = get_total_payoffs_all_agents(self, self.interm_payoff_vec_global, self.final_payoff_vec_global)[agent]
            self.payoff_dict_global["payoff_total"+str(agent)].append(total_payoff)
        
        if not experimental_mode:
            with open(os.path.join(path_to_results, self.name + ".txt"), 'a') as f:
                f.write(f"Environment trigger: {self.config.env_name}\n")
                for key, value in self.payoff_dict_global.items():
                    f.write(f"{key}: {value}\n")

        elif experimental_mode:
            # COLLECT AND SAVE DATA
            result_dict["runtime"] = end_time - start_time
            result_dict["winner"] = get_winner(self.mcts_nodes_global[-1].state)
            result_dict["T_terminal"] = self.mcts_nodes_global[-1].state.timestep
            result_dict["trajectories"] = [[float(value) for value in node.state.get_state_together()] for node in self.mcts_nodes_global]
            result_dict.update(self.payoff_dict_global)
            return result_dict
