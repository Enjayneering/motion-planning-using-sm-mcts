import os
import time
import pandas as pd

import matplotlib.pyplot as plt

from .bimatrix_mcts_objects import State, run_mcts
from .utilities.common_utilities import *
from .utilities.kinodynamic_utilities import *
from .utilities.payoff_utilities import *
from .utilities.csv_utilities import *
from .utilities.networkx_utilities import *
from .utilities.config_utilities import *
from .utilities.environment_utilities import *

from .shortest_path_planner import ShortestPathPlanner


class BiMatrixMCTS:
    def __init__(self, env, game_config, init_state=None):        
        self.config = game_config
        self.env = env
        if init_state is not None:
            # init state is list
            self.init_state = init_state
        else:       
            self.init_state = [self.env.init_state[0], self.env.init_state[1], self.env.init_state[2], self.env.init_state[3], self.env.init_state[4], self.env.init_state[5]]
            #[self.env.init_state['x0'], self.env.init_state['y0'], self.env.init_state['theta0'], self.env.init_state['x1'], self.env.init_state['y1'], self.env.init_state['theta1']]
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
        self.payoff_data_log["payoff_total"] = []
        
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
        "action_set_0": {"velocity": self.config.velocity_0,
                        "ang_velocity": self.config.ang_velocity_0},
        "action_set_1": {"velocity": self.config.velocity_1,
                        "ang_velocity": self.config.ang_velocity_1},
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
        # timestep sim, so that we can use the algorithm for simulation (only run for one stage the selfplay)

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
        if self.config.feature_flags["run_mode"]["exp"]:
            # initialize result dictionary
            result_dict = {}
            precomputed_traj = None
        
        policy_dict = {}
        policy_df = pd.DataFrame()

        # RUN TRAJECTORY PLANNER
        start_time = time.time()
        max_timestep = self.config.max_timehorizon

        # Precompute set of trajectories
        glob_traj_dict = {0: None, 1: None}
        trajectories = get_trajectories(self)
        
        """for agent in [0,1]:
            if self.config['predef_traj'][agent]['active']:
                num_traj = self.config['predef_traj'][agent]['num_traj']
                SPPlanner = ShortestPathPlanner(self.env, action_set=self.Kinodynamic_params['action_set_{}'.format(agent)], start_state=current_state_obj.get_state(agent=agent), start_timestep=current_state_obj.timestep, ix_agent=agent, dt=self.Model_params["delta_t"])
                traj_dict = SPPlanner.get_trajectories(num_trajectories=num_traj)
                trajectories[agent] = traj_dict['actions']
                glob_traj_dict[agent] = traj_dict
                #print("Trajectories for agent {}: {}".format(agent, trajectories[agent]))
                print(f"\n {len(trajectories[agent])} trajectories found for agent {agent}!\n")
            """

        while not is_terminal(self, current_state_obj, max_timestep=self.config.max_timehorizon) and (timesteps_sim is None or current_state_obj.timestep < timesteps_sim):
            print("Searching game tree in timestep {}...".format(current_state_obj.timestep))
            if self.config.feature_flags["run_mode"]["test"]:
                csv_init_rollout_last(self)

            print("Max timehorizon: {}".format(max_timestep))
             
            # RUN SINGLE MCTS ALGORITHM IN CURRENT TIMESTEP  
            next_state_obj, runtime, policies, traj_chosen = run_mcts(self, current_state_obj, max_timestep=max_timestep, trajectories=trajectories)

            trajectories = {0: [traj_chosen[0]],
                            1: [traj_chosen[1]]}
            
            #### flatten policy data to data frame
            for agent, policy_dict in policies.items():
                for action, policy_data in policy_dict.items():
                    new_row = {}
                    new_row["agent"] = agent
                    for key, value in policy_data.items():
                        new_row[key] = value
                    # round action and to string
                    new_row["action"] = str([round(value, 2) for value in new_row["action"]])
                    row_df = pd.DataFrame(new_row, index=[0])
                    policy_df = pd.concat([policy_df, row_df], ignore_index=True)

            if self.config.feature_flags["run_mode"]["exp"]:
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
            game_length = max_timestep-next_state_obj.timestep
            policy_dict[game_length] = policies

            if self.config.feature_flags["run_mode"]["test"]:
                csv_write_global_state(self, self.global_states[-1])
            
            self.global_states.append(next_state_obj)
            current_state_obj = next_state_obj

            if self.config.feature_flags["run_mode"]["exp"]:
                result_dict['game_length'] = game_length
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
        self.payoff_data_log["payoff_total"].append(total_payoff_list)

        if self.config.feature_flags["run_mode"]["test"]:
            with open(os.path.join(path_to_results, self.name + ".txt"), 'a') as f:
                f.write(f"Environment trigger: {self.config.name}\n")
                for key, value in self.payoff_data_log.items():
                    f.write(f"{key}: {value}\n")

        if self.config.feature_flags["run_mode"]["exp"]:
            # COLLECT AND SAVE DATA
            result_dict["winner"] = get_winner(self, self.global_states[-1])
            result_dict["runtime"] = end_time - start_time
            result_dict["T_terminal"] = self.global_states[-1].timestep
            result_dict["trajectory_0"] = [[float(value) for value in state.get_state(agent=0)]+[state.timestep] for state in self.global_states]
            result_dict["trajectory_1"] = [[float(value) for value in state.get_state(agent=1)]+[state.timestep] for state in self.global_states]
            result_dict.update(self.payoff_data_log) # merge dictionaries
            return result_dict, policy_df, glob_traj_dict
    


def get_trajectories(Game):
    trajectories = {0: None, 1: None}
    # get set of random trajectories in environment

    if Game.config['predef_traj'][1]['active']:
        trajectories[1] = []

        spin_left_0 = [0.0, -1.57]
        spin_right_0 = [0.0, 1.57]
        straight_0 = [1.0, 0.0]
        left_0 = [1.0, -1.57]
        right_0 = [1.0, 1.57]

        turn_left_small_0 = [left_0, right_0, straight_0]
        turn_right_small_0 = [right_0, left_0, straight_0]
        turn_left_large_0 = [left_0, straight_0, right_0]
        turn_right_large_0 = [right_0, straight_0, left_0]
        go_straight_0 = [straight_0, straight_0, straight_0]
        action_segments_dict_0 = { 'turn_left_small_0': turn_left_small_0, 
                            'turn_right_small_0': turn_right_small_0, 
                            'turn_left_large_0': turn_left_large_0, 
                            'turn_right_large_0': turn_right_large_0, 
                            'go_straight_0': go_straight_0,}

       

        # in street environment game is over after 
        keys = list(action_segments_dict_0.keys())
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i == j and ("large" in keys[i]):
                    continue  # Skip if both are large turns in the same direction
                if i == j and keys[i].count("small") > 1:
                    continue  # Skip if both are more than one small turn in the same direction
                if "left_small" in keys[i] and "left_large" in keys[j] or "left_large" in keys[i] and "left_small" in keys[j]:
                    continue
                if "right_small" in keys[i] and "right_large" in keys[j] or "right_large" in keys[i] and "right_small" in keys[j]:
                    continue
                trajectory = action_segments_dict_0[keys[i]] + action_segments_dict_0[keys[j]]
                trajectories[1].append(trajectory)
                #trajectories[0].append([spin_left_0] + trajectory)  # Add spin left at the beginning
                #trajectories[0].append([spin_right_0] + trajectory)  # Add spin right at the beginning
    
    if Game.config['predef_traj'][0]['active']:
        trajectories[0] = []
        
        spin_left_1 = [0.0, -1.57]
        spin_right_1 = [0.0, 1.57]
        
        straight_1 = [2.0, 0.0]
        left_1 = [2.0, -1.57]
        right_1 = [2.0, 1.57]

        turn_left_1 = [left_1, right_1, straight_1]
        turn_right_1 = [right_1, left_1, straight_1]
        go_straight_1 = [straight_1, straight_1, straight_1]
        action_segments_dict_1 = { 'turn_left_1': turn_left_1, 
                            'turn_right_1': turn_right_1, 
                            'go_straight_1': go_straight_1,}
  
        keys = list(action_segments_dict_1.keys())
        for i in range(len(keys)):
            for j in range(len(keys)):
                if i == j and ("turn" in keys[i]):
                    continue  # Skip if both are turns in the same direction
                trajectory = action_segments_dict_1[keys[i]] + action_segments_dict_1[keys[j]]
                trajectories[0].append(trajectory)
                #trajectories[1].append([spin_left_1] + trajectory)  # Add spin left at the beginning
                #trajectories[1].append([spin_right_1] + trajectory)  # Add spin right at the beginning

    # plot action segments for each agent
    #plt.plot(trajectories[0])
    #plt.show()
    #plt.plot(trajectories[1])
    #plt.show()
    
    """for agent in [0,1]:
        if game.config['predef_traj'][agent]['active']:
            trajectories[agent] = []
            num_traj = self.config['predef_traj'][agent]['num_traj']
            curr_timestep = current_state_obj.timestep
            for i in range(num_traj):
                trajectory = []
                traj_state_obj = current_state_obj
                for timestep in range(curr_timestep, max_timestep):
                    actions_0, actions_1 = sample_legal_actions(self, traj_state_obj)

                    action_0 = random.choice(actions_0)
                    action_1 = random.choice(actions_1)
                    actions = [action_0, action_1]
                    next_state_0 = mm_unicycle(traj_state_obj.get_state(agent=0), action=action_0, delta_t=self.Model_params["delta_t"])
                    next_state_1 = mm_unicycle(traj_state_obj.get_state(agent=1), action=action_1, delta_t=self.Model_params["delta_t"])
                    traj_state_obj = State(x0=next_state_0[0], y0=next_state_0[1], theta0=next_state_0[2], x1=next_state_1[0], y1=next_state_1[1], theta1=next_state_1[2], timestep=timestep+self.Model_params["delta_t"])
                    trajectory.append(actions[agent])
                trajectories[agent].append(trajectory)"""

    return trajectories