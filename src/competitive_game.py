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
        self.Kinodynamic_params = self._set_kinodynamic_params()

        self.forbidden_states = []
        self.global_states = [self._init_state()] # stores all nodes of the tree, represents the final trajectory

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
            "collision_distance": self.config.collision_distance,
            "agents": [0, 1],
            "interm_payoffs": {
                "penalty_distance_0": {"pos": 0, "weight": self.config.collision_ignorance*self.config.penalty_distance, "agent": 0},
                "penalty_distance_1": {"pos": 1, "weight": (1-self.config.collision_ignorance)*self.config.penalty_distance, "agent": 1},
                "reward_progress_0": {"pos": 2, "weight": self.config.reward_progress_0, "agent": 0},
                "reward_progress_1": {"pos": 3, "weight": self.config.reward_progress_1, "agent": 1},
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
    
    def _set_kinodynamic_params(self):
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
        }
        return MCTS_params
        
    def _init_state(self):
        # initialize root node
        initial_state = State(x0=self.env.init_state['x0'], 
                            y0=self.env.init_state['y0'], 
                            theta0=self.env.init_state['theta0'],
                            x1=self.env.init_state['x1'], 
                            y1=self.env.init_state['y1'], 
                            theta1=self.env.init_state['theta1'],
                            timestep=self.env.init_state['timestep'])
        return initial_state
    
    def _init_payoff_dict_global(self):
        # initialize payoff list
        payoff_dict_global = {}
        for payoff in self.Model_params["interm_payoffs"].keys():
            payoff_dict_global[payoff]= []
        for payoff in self.Model_params["final_payoffs"].keys():
            payoff_dict_global[payoff]= []
        for agent in self.Model_params["agents"]:
            payoff_dict_global["payoff_total_"+str(agent)] = []
        return payoff_dict_global
    
    def search_game_tree(self, global_state, game_max_timehorizon):
        start_time = time.time()

        # create Node for local tree search
        init_state = State(x0=global_state.x0,
                            y0=global_state.y0,
                            theta0=global_state.theta0,
                            x1=global_state.x1,
                            y1=global_state.y1,
                            theta1=global_state.theta1,
                            timestep=0)
        current_node = MCTSNode(Game=self, state=init_state)
        
        # TREE POLICY
        for iter in range(self.MCTS_params['num_iter']):
            #print("Horizon {} | Iteration {}".format(simhorizon, iter))
            v = current_node._tree_policy(self)

            #print("Starting rollout")
            rollout_trajectory, interm_payoff_rollout_vec, final_payoff_rollout_vec = v.rollout(self, game_max_timehorizon)
            
            payoff_total_list = get_total_payoffs_all_agents(self, interm_payoff_rollout_vec, final_payoff_rollout_vec)
            self.max_payoff, self.min_payoff, self.payoff_range = update_payoff_range(self.max_payoff, self.min_payoff, payoff_total_list)

            # write every x rollout trajectories
            if not experimental_mode:
                if iter % freq_stat_data == 0:
                    csv_write_rollout_last(self, rollout_trajectory, timehorizon = current_node.state.timestep, config=self.config)
            
            #print("Backpropagating")
            v.backpropagate(self, payoff_total_list)

            if is_feature_active(self.config.feature_flags["payoff_weights"]["adaptive"]):
                #self.interm_weights_vec = update_interm_weights(self, self.interm_weights_vec, interm_payoff_rollout_vec)
                #self.final_weights_vec = update_final_weights(self, self.final_weights_vec, final_payoff_rollout_vec)
                #payoff_weights = update_weigths_payoff(current_node, payoff_weights)
                pass
        
        next_node = current_node.select_final_child(Game=self)

        if not experimental_mode:
            save_tree_to_file(current_node, path_to_tree.format(global_state.timestep))
            #print("Next State: {}".format(next_node.state.get_state_together()))
        
        runtime = time.time() - start_time 

        # transfer search node to global state
        next_global_state = State(x0=next_node.state.x0,
                                y0=next_node.state.y0,
                                theta0=next_node.state.theta0,
                                x1=next_node.state.x1,
                                y1=next_node.state.y1,
                                theta1=next_node.state.theta1,
                                timestep=global_state.timestep+1)
        return next_global_state, runtime
    
    def compute_trajectories(self):
        # create a text trap and redirect stdout
        #text_trap = io.StringIO()
        #sys.stdout = text_trap

        current_state = self.global_states[0]

        if not experimental_mode:
            csv_init_global_state(self)
            init_tree_file()   
            csv_write_global_state(self, current_state)
            test_write_params(self)
        elif experimental_mode:
            # initialize result dictionary
            result_dict = {}

        # RUN TRAJECTORY PLANNER
        start_time = time.time()

        while not is_terminal(self, current_state, max_timehorizon=get_max_timehorizon(self.config)):
            print("Searching game tree in timestep {}...".format(current_state.timestep))
            if not experimental_mode:
                csv_init_rollout_last(self)
                
            if is_feature_active(self.config.feature_flags["payoff_weights"]["adaptive"]):
                self.interm_weights_vec = init_interm_weights(self)
                self.final_weights_vec = init_final_weights(self)
            elif is_feature_active(self.config.feature_flags["payoff_weights"]["fixed"]):
                self.interm_weights_vec = init_interm_weights(self)
                self.final_weights_vec = init_final_weights(self)
            
            # RUN SINGLE MCTS ALGORITHM IN CURRENT TIMESTEP
            game_max_timehorizon = get_max_timehorizon(self.config)-current_state.timestep
            next_state, runtime = self.search_game_tree(current_state, game_max_timehorizon)

            if experimental_mode:
                result_dict["runtime_game_length_{}".format(get_max_timehorizon(self.config)-current_state.timestep)] = runtime

            interm_payoff_increment_vec = get_intermediate_payoffs(self, current_state, next_state)
            self.interm_payoff_vec_global += interm_payoff_increment_vec

            # update global payoff dict
            for payoff in self.Model_params["interm_payoffs"].keys():
                self.payoff_dict_global[payoff].append(float(interm_payoff_increment_vec[self.Model_params["interm_payoffs"][payoff]["pos"]]))
            for agent in self.Model_params["agents"]:
                total_payoff = get_total_payoffs_all_agents(self, self.interm_payoff_vec_global, self.final_payoff_vec_global)[agent]
                self.payoff_dict_global["payoff_total_"+str(agent)].append(total_payoff)
            print("Payoff dict: {}".format(self.payoff_dict_global))
            if not experimental_mode:
                csv_write_global_state(self, self.global_states[-1])
            
            self.global_states.append(next_state)
            current_state = next_state
        
        if not experimental_mode:
                csv_write_global_state(self, self.global_states[-1])
        #print("Terminal state: {}".format(current_state.get_state_together()))
        
        # update final payoffs global solution
        final_payoff_vec = get_final_payoffs(self, current_state)    
        self.final_payoff_vec_global += final_payoff_vec

        end_time = time.time()
        print("Runtime: {} s".format(end_time - start_time))

        # update global payoff dict
        for payoff in self.Model_params["final_payoffs"].keys():
            self.payoff_dict_global[payoff].append(float(final_payoff_vec[self.Model_params["final_payoffs"][payoff]["pos"]]))
        for agent in self.Model_params["agents"]:
            total_payoff = get_total_payoffs_all_agents(self, self.interm_payoff_vec_global, self.final_payoff_vec_global)[agent]
            self.payoff_dict_global["payoff_total_"+str(agent)].append(total_payoff)
        print("Payoff dict: {}".format(self.payoff_dict_global))

        if not experimental_mode:
            with open(os.path.join(path_to_results, self.name + ".txt"), 'a') as f:
                f.write(f"Environment trigger: {self.config.instance_name}\n")
                for key, value in self.payoff_dict_global.items():
                    f.write(f"{key}: {value}\n")

        elif experimental_mode:
            # COLLECT AND SAVE DATA
            result_dict["winner"] = get_winner(self.global_states[-1])
            result_dict["runtime"] = end_time - start_time
            result_dict["T_terminal"] = self.global_states[-1].timestep
            result_dict["trajectory_0"] = [[float(value) for value in state.get_state_0()]+[state.timestep] for state in self.global_states]
            result_dict["trajectory_1"] = [[float(value) for value in state.get_state_1()]+[state.timestep] for state in self.global_states]
            result_dict.update(self.payoff_dict_global) # merge dictionaries
            return result_dict
