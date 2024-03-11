import os
import time

from .bimatrix_mcts_objects import State, run_mcts
from .utilities.common_utilities import *
from .utilities.kinodynamic_utilities import *
from .utilities.payoff_utilities import *
from .utilities.csv_utilities import *
from .utilities.networkx_utilities import *
from .utilities.config_utilities import *
from .utilities.environment_utilities import *


class BiMatrixMCTS:
    def __init__(self, env, game_config, init_state=None):        
        self.config = game_config
        self.env = env
        if init_state is not None:
            # init state is list
            self.init_state = init_state
        else:       
            self.init_state = [self.env.init_state['x0'], self.env.init_state['y0'], self.env.init_state['theta0'], self.env.init_state['x1'], self.env.init_state['y1'], self.env.init_state['theta1']]
        self.terminal_state = [self.env.goal_state['x0'], self.env.goal_state['y0'], self.env.goal_state['theta0'], self.env.goal_state['x1'], self.env.goal_state['y1'], self.env.goal_state['theta1']]
        self.name = get_next_game_name(path_to_results, game_config.name)

        self.MCTS_params = self._set_mcts_params()
        self.Model_params = self._set_model_params()
        self.Kinodynamic_params = self._set_kinodynamic_params()

        self.config['max_timehorizon'] = get_max_timehorizon(self)

        self.goal_state = self._get_goal_state()
        self.forbidden_states = []
        self.global_states = [self._init_state()] # stores all nodes of the tree, represents the final trajectory
        
        self.interm_payoff_list_global = []
        self.final_payoff_list_global = []
        self.payoff_data_log = {} # stores all payoff data for each timestep of the horizon

    def _set_model_params(self):
        Model_params = {
            "state_space": ['x0', 'y0', 'theta0', 'x1', 'y1', 'theta1', 'timestep'],
            "delta_t": self.config.delta_t,
            "collision_distance": self.config.collision_distance,
            "agents": [0, 1],
            }
        return Model_params
    

    def _get_goal_state(self):
        goal_state = State(x0=self.env.goal_state['x0'],
                            y0=self.env.goal_state['y0'],
                            theta0=self.env.goal_state['theta0'],
                            x1=self.env.goal_state['x1'],
                            y1=self.env.goal_state['y1'],
                            theta1=self.env.goal_state['theta1'],
                            timestep=get_max_timehorizon(self))
        return goal_state
    
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
        initial_state = State(x0=self.init_state[0],
                            y0=self.init_state[1],
                            theta0=self.init_state[2],
                            x1=self.init_state[3],
                            y1=self.init_state[4],
                            theta1=self.init_state[5],
                            timestep=0)
        return initial_state
    
    
    def run_game(self, timesteps_sim=None):
        # create a text trap and redirect stdout
        #text_trap = io.StringIO()
        #sys.stdout = text_trap

        # RUN A FULL SIMULATION OF A COMPETITIVE RACING GAME

        current_state_obj = self.global_states[0]

        if self.config.feature_flags["run_mode"]["test"]:
            csv_init_global_state(self)
            init_tree_file()   
            csv_write_global_state(self, current_state_obj)
            test_write_params(self)
        elif self.config.feature_flags["run_mode"]["exp"]:
            # initialize result dictionary
            result_dict = {}
        policy_dict = {}

        # RUN TRAJECTORY PLANNER
        start_time = time.time()
        max_timestep = self.config.max_timehorizon

        while not is_terminal(self, current_state_obj, max_timestep=self.config.max_timehorizon) and (timesteps_sim is None or current_state_obj.timestep < timesteps_sim):
            print("Searching game tree in timestep {}...".format(current_state_obj.timestep))
            if self.config.feature_flags["run_mode"]["test"]:
                csv_init_rollout_last(self)

            print("Max timehorizon: {}".format(max_timestep))
            # RUN SINGLE MCTS ALGORITHM IN CURRENT TIMESTEP  
            next_state_obj, runtime, policies = run_mcts(self, current_state_obj, max_timestep=max_timestep)
            

            if self.config.feature_flags["run_mode"]["exp"]:
                result_dict["alphat_eff_gamelength_{}".format(max_timestep)] = max_timestep/get_min_time_to_complete(self, curr_state=current_state_obj.get_state_together())
                result_dict["runtime_gamelength_{}".format(max_timestep)] = runtime

            interm_payoff_sum_incr, interm_payoff_each_incr = get_intermediate_payoffs(self, current_state_obj, next_state_obj)
            self.interm_payoff_list_global.append(interm_payoff_sum_incr)

            # UPDATE DATA LOG
            for key, value in interm_payoff_each_incr.items():
                if self.payoff_data_log.get(key) is None:
                    self.payoff_data_log[key] = []
                    self.payoff_data_log[key].append(value)
                else:
                    self.payoff_data_log[key].append(value)

            total_payoff_list = get_total_payoffs(self, self.interm_payoff_list_global, self.final_payoff_list_global)
            print("Total payoff list: {}".format(total_payoff_list))

            # Add total payoff to global payoff log
            if self.payoff_data_log.get("payoff_total") is None:
                self.payoff_data_log["payoff_total"] = []
            self.payoff_data_log["payoff_total"].append(total_payoff_list)
            
            # Append agents policies at each timestep
            policy_dict[max_timestep] = policies

            if self.config.feature_flags["run_mode"]["test"]:
                csv_write_global_state(self, self.global_states[-1])
            
            self.global_states.append(next_state_obj)
            current_state_obj = next_state_obj

            if self.config.feature_flags["run_mode"]["exp"]:
                result_dict['game_length'] = max_timestep-next_state_obj.timestep
            #max_timestep -= 1
        print("Terminal state: {}".format(current_state_obj.get_state_together()))
        print("Timestep: {}".format(current_state_obj.timestep))
    
        if self.config.feature_flags["run_mode"]["test"]:
                csv_write_global_state(self, self.global_states[-1])
        #print("Terminal state: {}".format(current_state_obj.get_state_together()))
        
        # FINAL PAYOFF
        final_payoff_sum_incr, final_payoff_each_incr = get_final_payoffs(self, current_state_obj)    
        self.final_payoff_list_global.append(final_payoff_sum_incr)

        end_time = time.time()
        print("Runtime: {} s".format(end_time - start_time))

        # update global payoff log
        for key, value in final_payoff_each_incr.items():
            if self.payoff_data_log.get(key) is None:
                self.payoff_data_log[key] = []
                self.payoff_data_log[key].append(value)
            else:
                self.payoff_data_log[key].append(value)

        total_payoff_list = get_total_payoffs(self, self.interm_payoff_list_global, self.final_payoff_list_global)

        # Add total payoff to global payoff log
        self.payoff_data_log["payoff_total"][-1] = total_payoff_list

        if self.config.feature_flags["run_mode"]["test"]:
            with open(os.path.join(path_to_results, self.name + ".txt"), 'a') as f:
                f.write(f"Environment trigger: {self.config.name}\n")
                for key, value in self.payoff_data_log.items():
                    f.write(f"{key}: {value}\n")

        elif self.config.feature_flags["run_mode"]["exp"]:
            # COLLECT AND SAVE DATA
            result_dict["winner"] = get_winner(self, self.global_states[-1])
            result_dict["runtime"] = end_time - start_time
            result_dict["T_terminal"] = self.global_states[-1].timestep
            result_dict["trajectory_0"] = [[float(value) for value in state.get_state(agent=0)]+[state.timestep] for state in self.global_states]
            result_dict["trajectory_1"] = [[float(value) for value in state.get_state(agent=1)]+[state.timestep] for state in self.global_states]
            result_dict.update(self.payoff_data_log) # merge dictionaries
            return result_dict, policy_dict
    
