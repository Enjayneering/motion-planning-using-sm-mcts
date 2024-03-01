import os
import time

# import modules
import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS')
from mcts import *
from agent import *
from utilities.common_utilities import *
from environment import *
from utilities.kinodynamic_utilities import *
from utilities.payoff_utilities import *
from utilities.csv_utilities import *
from utilities.networkx_utilities import *

#TODO: change

class Simulation:
    def __init__(self, config_global):        
        self.config_global = config_global
        self.env_global = Environment(config_global)
        self.name = get_next_game_name(path_to_results, self)

        #self.MCTS_params = self._set_mcts_params()
        #self.Model_params = self._set_model_params()
        #self.Kinodynamic_params = self._set_kinodynamic_params()

        # initialize agents
        self.agents = {}
        

        #add max timehorizon to config_global
        self.game_timehorizon = self.env_global.get_game_timehorizon(self.config_global)
        self.max_timesteps_sim = config_global['config_research']['max_timesteps_sim']
        self.config_global['game_timehorizon'] = self.game_timehorizon
        
        self.forbidden_states = []
        self.global_trajectory = {'states': [],
                                  'timesteps': []} # stores all taken states and timesteps
        
        self.payoff_data_log = {} # stores all payoff data for each timestep of the horizon

    """def _set_model_params(self):
        Model_params = {
            "state_space": ['x0', 'y0', 'theta0', 'x1', 'y1', 'theta1', 'timestep'],
            "agents": [0, 1],
            }
        return Model_params"""
    

    """def _get_goal_state(self):
        goal_state = State(x0=self.env.goal_state['x0'],
                            y0=self.env.goal_state['y0'],
                            theta0=self.env.goal_state['theta0'],
                            x1=self.env.goal_state['x1'],
                            y1=self.env.goal_state['y1'],
                            theta1=self.env.goal_state['theta1'],
                            timestep=get_max_timehorizon(self))
        return goal_state"""
    
    """def _set_kinodynamic_params(self):
        Kinodynamic_params = {
        "action_set_0": {"velocity_0": self.config.velocity_0,
                        "ang_velocity_0": self.config.ang_velocity_0},
        "action_set_1": {"velocity_1": self.config.velocity_1,
                        "ang_velocity_1": self.config.ang_velocity_1},
        }
        return Kinodynamic_params"""

    """def _set_mcts_params(self):
        MCTS_params = {
        "num_iter": self.config.num_iter, #max number of simulations, proportional to complexity of the game
        "c_param": self.config.c_param, # c_param: exploration parameter | 3.52 - Tuned from Paper by Perick, 2012
        }
        return MCTS_params"""
        
    """def _init_state(self):
        # initialize root node
        initial_state = State(x0=self.env.init_state['x0'], 
                            y0=self.env.init_state['y0'], 
                            theta0=self.env.init_state['theta0'],
                            x1=self.env.init_state['x1'], 
                            y1=self.env.init_state['y1'],
                            theta1=self.env.init_state['theta1'],
                            timestep=0)
        return initial_state"""

    def run_simulation(self, test_id):
        # create a text trap and redirect stdout
        #text_trap = io.StringIO()
        #sys.stdout = text_trap

        # RUN A FULL SIMULATION WITHIN THE ENVIRONMENT

        # INITIALIZE
        start_time = time.time()
        
        curr_state_sim = self.env_global.joint_init_state
        curr_timestep_sim = 0
        self.global_trajectory['states'].append(curr_state_sim)
        self.global_trajectory['timesteps'].append(curr_timestep_sim)

        """if self.config_global['config_research']["run_mode"]["test"]:
            csv_init_global_state(test_id, self.config_global)
            csv_write_global_state(test_id, self.config_global, curr_state_sim)"""

        # create agents and initialize simulation
        for agent_ix in self.config_global['config_agents'].keys():
            self.agents[agent_ix] = Agent(agent_ix, curr_timestep_sim, self.game_timehorizon, self.config_global['config_agents'], self.config_global['config_environment'])
            
            """if self.config_global['config_research']["run_mode"]["test"]:
                init_tree_file(test_id, path_to_trees)   
                write_config_txt(self.config_global)"""
            
            if self.config_global['config_research']["run_mode"]["exp"]:
                # initialize result dictionary
                for agent_ix, agent in self.agents.items():
                    result_dict = {}
                    result_dict[agent_ix] = {}
            policy_dict = {}
            policy_dict[agent_ix] = {}
        
        #TODO: global param?
        game_timehorizon = self.config_global['max_timehorizon']

        # RUN SIMULATION LOOP
        while not is_terminal(self.env, curr_state_sim, curr_timestep_sim, game_timehorizon=game_timehorizon) and (self.max_timesteps_sim is None or curr_timestep_sim < self.max_timesteps_sim):
            print("Searching game tree in timestep {}...".format(curr_timestep_sim))
            
            if self.config_global['config_research']["run_mode"]["test"]:
                csv_init_rollout_last(self.config_global)

            print("game_timehorizon: {}".format(game_timehorizon))

            # COMPUTE NEXT STEP, EACH AGENT ON HIS OWN    
            next_joint_state = np.zeros((len(self.agents), len(self.config_global['config_environment']['state_space'])))
            
            for agent_ix, agent in self.agents.items():
                # if closed loop, update agents states         
                if self.config_global['config_research']["sensing"]["closed"]:
                    agent.update_current_joint_state(curr_state_sim)
                
                next_joint_state_agent, runtime_agent, policies_agent = agent.compute_next_step(game_timehorizon)
                # take agents own state (first element) and sort it in global state
                next_joint_state[agent_ix] = next_joint_state_agent[0]

                if self.config_global['config_research']["run_mode"]["exp"]:
                    result_dict[agent_ix]["runtime_gamelength_{}".format(self.game_timehorizon)] = runtime_agent
                    result_dict[agent_ix]["alphat_eff_gamelength_{}".format(self.game_timehorizon)] = self.get_alpha_t_eff(curr_state_sim)
                # Append agents policies at each timestep
                policy_dict[agent_ix][self.game_timehorizon] = policies_agent
            
            next_timestep = curr_timestep_sim + 1
            
            """# UPDATE DATA LOG
            for key, value in interm_payoff_each_incr.items():
                if Game.payoff_data_log.get(key) is None:
                    Game.payoff_data_log[key] = []
                    Game.payoff_data_log[key].append(value)
                else:
                    Game.payoff_data_log[key].append(value)"""

            """# Add total payoff to global payoff log
            if self.payoff_data_log.get("payoff_total") is None:
                self.payoff_data_log["payoff_total"] = []
            self.payoff_data_log["payoff_total"].append(total_payoff_list)"""
            
            """if self.config_global['config_research']["run_mode"]["test"]:
                csv_write_global_state(Game, Game.global_states[-1])"""
            
            self.global_trajectory['states'].append(next_joint_state)
            self.global_trajectory['timesteps'].append(next_timestep)
            curr_state_sim = next_joint_state

            if self.config_global['config_research']["run_mode"]["exp"]:
                result_dict['game_timehorizon'] = game_timehorizon
            game_timehorizon -= 1

        """if self.config_global['config_research']["run_mode"]["test"]:
                csv_write_global_state(test_id, self.config_global, curr_state_sim)"""
        
        #print("Terminal state: {}".format(current_state_obj.get_state_together()))

        end_time = time.time()
        print("Runtime: {} s".format(end_time - start_time))

        # update global payoff log
        """for key, value in final_payoff_each_incr.items():
            if Game.payoff_data_log.get(key) is None:
                Game.payoff_data_log[key] = []
                Game.payoff_data_log[key].append(value)
            else:
                Game.payoff_data_log[key].append(value)"""

        # Add total payoff to global payoff log
        """Game.payoff_data_log["payoff_total"][-1] = total_payoff_list"""

        """if self.config_global['config_research']["run_mode"]["test"]:
            with open(os.path.join(path_to_results, Game.name + ".txt"), 'a') as f:
                f.write(f"Environment trigger: {Game.config.env_name}\n")
                for key, value in Game.payoff_data_log.items():
                    f.write(f"{key}: {value}\n")"""
        
        # Calculate payoff based on trajectory
        total_payoff, self.payoff_data_log = get_total_payoff()

        if self.config_global['config_research']["run_mode"]["exp"]:
            # COLLECT AND SAVE DATA
            result_dict["winner"] = get_winner(self)
            result_dict["runtime"] = end_time - start_time
            result_dict["T_terminal"] = self.global_trajectory[-1]['timestep']
            result_dict["trajectory"] = self.global_trajectory
            result_dict.update(self.payoff_data_log) # merge dictionaries
            return result_dict, policy_dict

    def get_alpha_t_eff(self, curr_state_sim):
        return self.game_timehorizon/self.env_global.get_min_time_to_complete(curr_state_sim, self.env_global.joint_terminal_state, self.config_global)