import io
import sys

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
        self.Competitive_params = self._set_competitive_params()

        self.forbidden_states = []
        self._mcts_nodes = [self._init_root_node()]

        # store the final trajectory
        self.trajectory = [self._mcts_nodes[0].state.get_state_together()]

        self.max_payoff = 0
        self.min_payoff = 0
        self.payoff_range = 0
        self.interm_weights_vec = init_interm_weights(self)
        self.final_weights_vec = init_final_weights(self)
        

        self.payoff_dict_global = self._init_payoff_dict()
        self.interm_payoff_vec_global = np.zeros((self.Model_params["len_interm_payoffs"],1))
        self.final_payoff_vec_global = np.zeros((self.Model_params["len_final_payoffs"],1))

    def _set_model_params(self):
        Model_params = {
            "delta_t": 1,
            "agents": [0, 1],
            "state_space": ['x0', 'y0', 'theta0', 'x1', 'y1', 'theta1', 'timestep'],
            "action_space": ['x0', 'y0', 'x1', 'y1'],
            "interm_payoffs": {
                "penalty_collision_0": {"pos": 0, "weight": self.config.penalty_collision_0, "agent": 0},
                "penalty_collision_1": {"pos": 1, "weight": self.config.penalty_collision_1, "agent": 1},
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
    
    def _set_competitive_params(self):
        Competitive_params = {
        "action_set_0": {"velocity_0": np.linspace(0, 2, 3).tolist(),
                        "ang_velocity_0": np.linspace(-np.pi/2, np.pi/2, 3).tolist()},
        "action_set_1": {"velocity_1": np.linspace(0, 1, 2).tolist(),
                        "ang_velocity_1": np.linspace(-np.pi/2, np.pi/2, 3).tolist()},
        }
        return Competitive_params

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
    
    def _init_payoff_dict(self):
        # initialize payoff list
        payoff_dict_global = {}
        for agent in self.Model_params["agents"]:
            payoff_dict_global[agent] = [(0,0)]
        return payoff_dict_global
    
    def _update_payoff_dict(self, payoff_list):
        for agent in self.Model_params["agents"]:
            self.payoff_dict_global[agent].append((self._mcts_nodes[-1].state.timestep, payoff_list[agent]))
    
    def search_game_tree(self):
        node_current_timestep = self._mcts_nodes[-1]
        simhorizon = node_current_timestep.state.timestep
        
        for iter in range(self.MCTS_params['num_iter']):
            print("Horizon {} | Iteration {}".format(simhorizon, iter))
            #print("Starting tree policy")
            v = node_current_timestep._tree_policy(self)

            #print("Starting rollout")
            rollout_trajectory, interm_payoff_rollout, final_payoff_rollout = v.rollout(self)
            payoff_list = get_total_payoffs_all_agents(self, interm_payoff_rollout, final_payoff_rollout)
            self.max_payoff, self.min_payoff, self.payoff_range = update_payoff_range(self.max_payoff, self.min_payoff, payoff_list)
            

            # write every x rollout trajectories
            if iter % freq_stat_data == 0:
                csv_write_rollout_last(self, rollout_trajectory, timehorizon = node_current_timestep.state.timestep, config=self.config)
            
            #print("Backpropagating")
            v.backpropagate(self, payoff_list)
            #payoff_weights = update_weigths_payoff(node_current_timestep, payoff_weights)
            #print("Payoff weights: {}".format(payoff_weights))
        
        #selected_action = node_current_timestep.select_action(payoff_range)
        
        #next_node = node_current_timestep.select_child(payoff_range, self.forbidden_states)
        next_node = node_current_timestep.robust_child()
        self._mcts_nodes.append(next_node)

        save_tree_to_file(node_current_timestep, path_to_tree.format(node_current_timestep.state.timestep))
        print("Next State: {}".format(next_node.state.get_state_together()))
        #return next_node.state, #, next_node.parent_action
    
    def find_nash_strategies(self):
        # create a text trap and redirect stdout
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # initialize csv files
        csv_init_global_state(self)
        init_tree_file()   

        node_current_horizon = self._mcts_nodes[-1]

        csv_write_global_state(self, node_current_horizon.state)
        
        #save to text file
        if is_feature_active(feature_flags["mode"]["test"]):
            test_write_params(self)


        while not is_terminal(self.env, node_current_horizon.state):
            csv_init_rollout_last(self)

            self.interm_weights_vec = init_interm_weights(self)
            self.final_weights_vec = init_final_weights(self)

            self.search_game_tree()

            # update payoffs global solution
            self.interm_payoff_vec_global = update_intermediate_payoffs(self, self.interm_payoff_vec_global, node_current_horizon.state, self._mcts_nodes[-1].state)
            
            payoff_list = get_total_payoffs_all_agents(self, self.interm_payoff_vec_global, self.final_payoff_vec_global)
            
            self._update_payoff_dict(payoff_list)

            self.trajectory.append(self._mcts_nodes[-1].state.get_state_together())

            csv_write_global_state(self, self._mcts_nodes[-1].state)

            node_current_horizon = self._mcts_nodes[-1]
            
            self._mcts_nodes.append(node_current_horizon)

        # update final payoffs global solution
        self.final_payoff_vec_global = update_final_payoffs(self, self.final_payoff_vec_global, node_current_horizon.state)
        payoff_list = get_total_payoffs_all_agents(self, self.interm_payoff_vec_global, self.final_payoff_vec_global)
        self._update_payoff_dict(payoff_list)

        # save final payoffs to textfile
        if is_feature_active(feature_flags["mode"]["test"]):
            with open(os.path.join(path_to_results, self.name + ".txt"), 'a') as f:
                f.write(f"Environment trigger: {self.env.env_name_trigger}\n")
                for agent in self.Model_params["agents"]:
                    f.write(f"Payoff Agent {agent}: {self.payoff_dict_global[agent]}\n")
