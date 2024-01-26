import sys
import numpy as np
import itertools
import logging

from common import *
from csv_utilities import *
from payoff_utilities import *
from kinodynamic_utilities import *

logging.basicConfig(filename='mcts.log', encoding='utf-8', level=logging.DEBUG)

def roundme(number):
    return number #np.round(number, 2)

class State:
    def __init__(self, x0, y0, theta0, x1, y1, theta1, timestep):
        self.x0 = roundme(x0)
        self.y0 = roundme(y0)
        self.x1 = roundme(x1)
        self.y1 = roundme(y1)
        self.theta0 = roundme(theta0)
        self.theta1 = roundme(theta1)
        self.timestep = timestep

    def move(self, action, delta_t):
        # transition-function: get new state from previous state and chosen action
        state_0 = self.get_state_0()
        action_0 = action[:2]

        state_1 = self.get_state_1()
        action_1 = action[2:]

        x0_new, y0_new, theta0_new = mm_unicycle(state_0, action_0, delta_t=delta_t)
        x1_new, y1_new, theta1_new = mm_unicycle(state_1, action_1, delta_t=delta_t)

        timestep_new = self.timestep + delta_t
        return State(x0_new, y0_new, theta0_new, x1_new, y1_new, theta1_new, timestep_new)

    def get_state_0(self):
        state_list = [self.x0, self.y0, self.theta0]
        return state_list

    def get_state_1(self):
        state_list = [self.x1, self.y1, self.theta1]
        return state_list

    def get_state_together(self):
        state_list = [self.x0, self.y0, self.theta0, self.x1, self.y1, self.theta1, self.timestep]
        return state_list

class MCTSNode:
    def __init__(self, Game, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []

        ########## MCTS parameters ##########
        self._aggr_payoffs = [0]*len(Game.Model_params["agents"])
        self._number_of_visits = 1

        self._untried_actions = self.untried_actions(Game)
        self.select_policy_data = {"action_stats_0": {}, "action_stats_1": {}}
        #self.action_stats_0, self.action_stats_1 = None, None #self.init_action_stats() # [action, number_count, sum of payoffs]
        

    """def init_action_stats(self):
        legal_action_0, legal_action_1, _ = sample_legal_actions(Game, self.state)
        self.action_stats_0 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_0]
        self.action_stats_1 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_1]
        return self.action_stats_0, self.action_stats_1"""

    """def update_action_stats(self, Game):
        legal_action_0, legal_action_1, _ = sample_legal_actions(Game, self.state)
        self.action_stats_0 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_0]
        self.action_stats_1 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_1]

        # count at each node the number of children visited for the respective action and update that action's stats
        # TODO: for multiple agents
        for i, action_stat in enumerate(self.action_stats_0):
            #self.action_stats_0[i]["num_count"] = 1
            #self.action_stats_0[i]["sum_payoffs"] = 0

            for c in self.children:
                if c.parent_action[:2] == action_stat["action"]: # and action_stat["action"] != -np.inf:
                    self.action_stats_0[i]["num_count"] += 1
                    self.action_stats_0[i]["sum_payoffs"] += c.X(agent=0)

        for i, action_stat in enumerate(self.action_stats_1):
            #self.action_stats_1[i]["num_count"] = 1
            #self.action_stats_1[i]["sum_payoffs"] = 0

            for c in self.children:
                if c.parent_action[2:] == action_stat["action"]: # and action_stat["action"] != -np.inf:
                    self.action_stats_1[i]["num_count"] += 1
                    self.action_stats_1[i]["sum_payoffs"] += c.X(agent=1)
        pass
"""
    def expand(self, Game):
        action = self._untried_actions.pop(np.random.randint(len(self._untried_actions))) #pop random action out of the list

        next_state = self.state.move(action, delta_t=Game.Model_params["delta_t"])

        child_node = MCTSNode(Game, next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        #print("Child node: {}".format(child_node.state.get_state_together()))
        return child_node

    def untried_actions(self, Game):
        _, _, untried_actions = sample_legal_actions(Game, self.state)
        return untried_actions

    # TODO: vary Selection policy
    def calc_UCT(self, action_stat, payoff_range, c_param):
            X = action_stat["sum_payoffs"]
            n = action_stat["num_count"]
            UCT = (X/n) + c_param *payoff_range* np.sqrt((np.log(self.n()) / n)) # payoff range normalizes the payoffs so that we get better exploration
            return UCT
    
    def _select_action_UCT(self, Game, agent = 0):
        weights = [self.calc_UCT(action_stat, Game.payoff_range, Game.MCTS_params['c_param']) for action_stat in self.select_policy_data['action_stats_{}'.format(agent)].values()]
        action_select = list(self.select_policy_data['action_stats_{}'.format(agent)].values())[np.argmax(weights)]['action']
        return action_select

    """def select_action(self, Game):
        # update action stats based on all possible and already visited childs
        #print("Try to select action")
        
        #self.update_action_stats(Game)

        # UCT: returns the child with the highest UCB1 score
        # Note: we need to choose the most promising action, but ensuring that they are also collisionfree
        # TODO: Multi Agent adjustment
        weights_0 = [self.calc_UCT(action_stat, Game.payoff_range, Game.MCTS_params['c_param']) for action_stat in self.select_policy_data['action_stats_0'].values()]
        weights_1 = [self.calc_UCT(action_stat, Game.payoff_range, Game.MCTS_params['c_param']) for action_stat in self.select_policy_data['action_stats_1'].values()]
        
        #print("Weights 0: {}\nWeights 1: {}".format(weights_0, weights_1))
        #print("Action stats 0: {}\nAction stats 1: {}".format(self.select_policy_data['action_stats_0'], self.select_policy_data['action_stats_1']))

        # resolve selecting actions that lead to a collision
        while True:
            action_select_0 = list(self.select_policy_data['action_stats_0'].values())[np.argmax(weights_0)]['action']
            action_select_1 = list(self.select_policy_data['action_stats_1'].values())[np.argmax(weights_1)]['action']
            break
            x0, y0, theta0 = mm_unicycle(self.state.get_state_0(), action_select_0, delta_t=Game.Model_params["delta_t"])
            x1, y1, theta1 = mm_unicycle(self.state.get_state_1(), action_select_1, delta_t=Game.Model_params["delta_t"])
            if distance([x0, y0], [x1, y1]) < Game.Model_params["collision_distance"]:
                weights_0[np.argmax(weights_0)] = -np.inf
                weights_1[np.argmax(weights_1)] = -np.inf
                #print("Collision detected from state {} to state {}".format(self.state.get_state_together(), [x0, y0, theta0, x1, y1, theta1, self.state.timestep+Game.Model_params["delta_t"]]))
            else:
                break

        selected_action = action_select_0 + action_select_1

        #print("Selected action: {}".format(selected_action))
        
        return selected_action"""
    
    def _select_action_max(self, Game, agent = 0):
        weights = [action_stat["sum_payoffs"]/action_stat['num_count'] for action_stat in self.select_policy_data['action_stats_{}'.format(agent)].values()]
        action_select = list(self.select_policy_data['action_stats_{}'.format(agent)].values())[np.argmax(weights)]['action']
        return action_select
    
    def _select_action_robust(self, Game, agent = 0, num_child = 0):
        weights = [action_stat["num_count"] for action_stat in self.select_policy_data['action_stats_{}'.format(agent)].values()]
        if num_child == 0:
            action_select = list(self.select_policy_data['action_stats_{}'.format(agent)].values())[np.argmax(weights)]['action']
            return action_select
        elif num_child > 0:
            actions_select = []
            for n_child in range(0, num_child):
                action_select = list(self.select_policy_data['action_stats_{}'.format(agent)].values())[np.argmax(weights)]['action']
                weights[np.argmax(weights)] = -np.inf
                actions_select.append(action_select)
            return actions_select

    def select_policy_child(self, Game):
        # resolve selecting actions that lead to a collision
        if Game.config.feature_flags['collision_handling']['pruning']:
            if Game.config.feature_flags['selection_policy']['ucb']:
                weights_0 = [self.calc_UCT(action_stat, Game.payoff_range, Game.MCTS_params['c_param']) for action_stat in self.select_policy_data['action_stats_0'].values()]
                weights_1 = [self.calc_UCT(action_stat, Game.payoff_range, Game.MCTS_params['c_param']) for action_stat in self.select_policy_data['action_stats_1'].values()]
            elif Game.config.feature_flags['selection_policy']['greedy']:
                weights_0 = [action_stat["sum_payoffs"] for action_stat in self.select_policy_data['action_stats_0'].values()]
                weights_1 = [action_stat["sum_payoffs"] for action_stat in self.select_policy_data['action_stats_1'].values()]
            while True:
                selected_action_0 = list(self.select_policy_data['action_stats_0'].values())[np.argmax(weights_0)]['action']
                selected_action_1 = list(self.select_policy_data['action_stats_1'].values())[np.argmax(weights_1)]['action']
                x0, y0, theta0 = mm_unicycle(self.state.get_state_0(), selected_action_0, delta_t=Game.Model_params["delta_t"])
                x1, y1, theta1 = mm_unicycle(self.state.get_state_1(), selected_action_1, delta_t=Game.Model_params["delta_t"])
                if distance([x0, y0], [x1, y1]) < Game.Model_params["collision_distance"]:
                    weights_0[np.argmax(weights_0)] = -np.inf
                    weights_1[np.argmax(weights_1)] = -np.inf
                    #print("Collision detected from state {} to state {}".format(self.state.get_state_together(), [x0, y0, theta0, x1, y1, theta1, self.state.timestep+Game.Model_params["delta_t"]]))
                else:
                    break
        elif Game.config.feature_flags['collision_handling']['punishing']:
            if Game.config.feature_flags['selection_policy']['ucb']:
                selected_action_0 = self._select_action_UCT(Game, agent=0)
                selected_action_1 = self._select_action_UCT(Game, agent=1)
            elif Game.config.feature_flags['selection_policy']['max']:
                selected_action_0 = self._select_action_max(Game, agent=0)
                selected_action_1 = self._select_action_max(Game, agent=1)

        selected_action = selected_action_0 + selected_action_1
        #print( "Selected action: {}".format(selected_action))
        child = [child for child in self.children if child.parent_action == selected_action][0]
        #print("Selected child: {}".format(child.state.get_state_together()))
        return child

    def _robust_child_joint(self):
        choices_weights = [c.n() for c in self.children]
        index = np.argmax(choices_weights)
        robust_child = self.children[index]
        return robust_child
    
    def _pop_robust_child_from_list(self, children):
        choices_weights = [c.n() for c in children]
        index = np.argmax(choices_weights)
        robust_child = children.pop(index)
        max_value = choices_weights[index]
        return robust_child, max_value
        
    def _sum_children_weights(self, Game, parent_node, parent_weight, depth, num_children):
        # returning robust weight and calling function for robust child
        all_children = [] +parent_node.children # list
        for n_child in range(0, num_children):
            if depth <= Game.config.final_move_depth and len(all_children) > 0:
                robust_child, max_value = parent_node._pop_robust_child_from_list(all_children)
                parent_weight += max_value
                parent_weight = self._sum_children_weights(Game, robust_child, parent_weight, depth+1, num_children)
        return parent_weight
    
    def select_final_child(self, Game):
        if Game.config.feature_flags['final_move']['robust-joint']:
            child = self._robust_child_joint()
            print("Robust child joint: {}".format(child.state.get_state_together()))
        elif Game.config.feature_flags['final_move']['robust-separate']:
            selected_action_0 = self._select_action_robust(Game, agent=0)
            selected_action_1 = self._select_action_robust(Game, agent=1)
            selected_action = selected_action_0 + selected_action_1
            child = [child for child in self.children if child.parent_action == selected_action][0]
            print("Robust child separate: {}".format(child.state.get_state_together()))
        elif Game.config.feature_flags['final_move']['depth-robust-joint']:
            all_children = []+self.children # new list
            next_children = []
            next_weights = []
            if Game.config.num_final_move_childs == None:
                num_children = len(all_children)
            else:
                num_children = Game.config.num_final_move_childs
            for n_child in range(0, num_children):
                # initialize next candidate
                next_child, next_weight = self._pop_robust_child_from_list(all_children)
                next_children.append(next_child)
                # search through all childs until depth d
                next_weight = self._sum_children_weights(Game, next_child, next_weight, 0, num_children)
                next_weights.append(next_weight)
            for next_child in next_children:
                self.children.append(next_child)
            print("Robust depth weights: {}".format(next_weights))
            print("Best child index: {}".format(np.argmax(next_weights)))
            best_child_index = np.argmax(next_weights)
            child = next_children[best_child_index]
            print("Robust depth child: {}".format(child.state.get_state_together()))
        elif Game.config.feature_flags['final_move']['depth-robust-separate']:
            selected_actions_0 = self._select_action_robust(Game, agent = 0, num_child = Game.config.num_final_move_childs)
            selected_actions_1 = self._select_action_robust(Game, agent = 1, num_child = Game.config.num_final_move_childs)
            action_combinations_together = [selected_action_0+selected_action_1 for selected_action_0, selected_action_1 in zip(selected_actions_0, selected_actions_1)]
            #action_combinations_together = list(itertools.product(selected_actions_0, selected_actions_1))
            #action_combinations_together = [list(action_pair) for action_pair in action_combinations_together]
            #action_combinations_together = [action_pair[0] + action_pair[1] for action_pair in action_combinations_together]  
            children = []
            weights =[]
            for action_combination in action_combinations_together:
                child = [child for child in self.children if child.parent_action == action_combination][0]
                children.append(child)
                weights.append(child.n())
            best_child_index = np.argmax(weights)
            child = children[best_child_index]
        elif Game.config.feature_flags['final_move']['max']:
            selected_action_0 = self._select_action_max(Game, agent=0)
            selected_action_1 = self._select_action_max(Game, agent=1)
            selected_action = selected_action_0 + selected_action_1
            child = [child for child in self.children if child.parent_action == selected_action][0]
            print("Best child: {}".format(child.state.get_state_together()))
        elif Game.config.feature_flags['final_move']['ucb']:
            selected_action_0 = self._select_action_UCT(Game, agent=0)
            selected_action_1 = self._select_action_UCT(Game, agent=1)
            selected_action = selected_action_0 + selected_action_1
            child = [child for child in self.children if child.parent_action == selected_action][0]
            print("UCT child: {}".format(child.state.get_state_together()))
        print("Action Value Agent 0 at timestep {}: {}".format(self.state.timestep, child.parent.select_policy_data['action_stats_0'][str(child.parent_action[:2])]['sum_payoffs']/child.parent.select_policy_data['action_stats_0'][str(child.parent_action[:2])]['num_count']))
        print("Action Value Agent 1 at timestep {}: {}".format(self.state.timestep, child.parent.select_policy_data['action_stats_1'][str(child.parent_action[2:])]['sum_payoffs']/child.parent.select_policy_data['action_stats_1'][str(child.parent_action[2:])]['num_count']))
        return child

    def n(self):
        return self._number_of_visits

    #def X(self, agent=None):
        # get sum of payoffs for agent index
        """payoff_agent = 0
        for payoffs in Model_params["payoff_vector"].values():
            for payoff in payoffs.values():
                if payoff["agent"] == agent:
                    payoff_agent += float(self._payoff_vector[payoff["pos"]])"""
        return self._aggr_payoffs[agent]

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def rollout(self, Game):
        # rollout policy: random action selection
        current_rollout_node = self

        rollout_trajectory = [current_rollout_node.state]
        
        interm_payoff_vec = np.zeros((Game.Model_params["len_interm_payoffs"],1))
        final_payoff_vec = np.zeros((Game.Model_params["len_final_payoffs"],1))

        # intermediate timesteps
        while not is_terminal(Game, current_rollout_node.state):
            #print("Rollout State: {}".format(current_rollout_node.state.get_state_together()))
            moves_0, moves_1, possible_moves = sample_legal_actions(Game, current_rollout_node.state)

            #print("Moves 0: {}, Moves 1: {}".format(moves_0, moves_1))
            #print("Possible moves: {}".format(possible_moves))

            #TODO: change parameter punishment when stuck (or pruning..?)

            if len(moves_0) == 0 and len(moves_1) == 0:
                Game.forbidden_states.append(current_rollout_node.state.get_state_0()+[current_rollout_node.state.timestep])
                Game.forbidden_states.append(current_rollout_node.state.get_state_1()+[current_rollout_node.state.timestep])
                #payoff_0 += MCTS_params['penalty_stuck_in_env']
                #payoff_1 += MCTS_params['penalty_stuck_in_env']
                print("Both agents stuck in environment, append current state on forbidden list, break")
                break
            elif len(moves_0) == 0:
                Game.forbidden_states.append(current_rollout_node.state.get_state_0()+[current_rollout_node.state.timestep])
                #payoff_0 += MCTS_params['penalty_stuck_in_env']
                print("Agent 0 stuck in environment, append current state on forbidden list, break")
                break
            elif len(moves_1) == 0:
                Game.forbidden_states.append(current_rollout_node.state.get_state_1()+[current_rollout_node.state.timestep])
                #payoff_1 += MCTS_params['penalty_stuck_in_env']
                print("Agent 1 stuck in environment, append current state on forbidden list, break")
                break
            #print("Forbidden states: {}".format(Game.forbidden_states))

            # choose action due to rollout policy
            action = self.rollout_policy(Game, current_rollout_node, moves_0, moves_1, possible_moves)

            #print("Rollout Action: {}".format(action))
            next_rollout_state = current_rollout_node.state.move(action, delta_t=Game.Model_params["delta_t"])
            next_rollout_node = MCTSNode(Game, next_rollout_state, parent=current_rollout_node, parent_action=action)

            # updating intermediate payoffs
            interm_payoff_vec += get_intermediate_payoffs(Game, current_rollout_node.state, next_rollout_node.state, discount_factor=Game.config.discount_factor)

            current_rollout_node = next_rollout_node
            rollout_trajectory.append(current_rollout_node.state)

        # updating final payoffs
        if is_terminal(Game, current_rollout_node.state):
            final_payoff_vec += get_final_payoffs(Game, current_rollout_node.state)

        return rollout_trajectory, interm_payoff_vec, final_payoff_vec
    

    def rollout_policy(self, Game, current_rollout_node, moves_0, moves_1, possible_moves):
        if current_rollout_node.state.timestep <= Game.config.rollout_length:
            if Game.config.feature_flags['rollout_policy']['random-uniform']:
                # choose a random action to simulate rollout
                action = possible_moves[np.random.randint(len(possible_moves))]
            elif Game.config.feature_flags['rollout_policy']['random-informed']:
                # best reference action (heuristic)
                weights_angle_0 = [mm_unicycle(current_rollout_node.state.get_state_0(), action, delta_t=Game.config.delta_t)[2]%(np.pi) for action in moves_0]
                weights_angle_1 = [mm_unicycle(current_rollout_node.state.get_state_1(), action, delta_t=Game.config.delta_t)[2]%(np.pi) for action in moves_1]
                weights_dist_0 = [distance(mm_unicycle(current_rollout_node.state.get_state_0(), action, delta_t=Game.config.delta_t), [Game.config.finish_line, current_rollout_node.state.y0]) for action in moves_0]
                weights_dist_1 = [distance(mm_unicycle(current_rollout_node.state.get_state_1(), action, delta_t=Game.config.delta_t), [Game.config.finish_line, current_rollout_node.state.y1]) for action in moves_1]
                action_0 = moves_0[np.argmin(np.add(weights_angle_0, weights_dist_0))]
                action_1 = moves_1[np.argmin(np.add(weights_angle_1, weights_dist_1))]

                # Sample velocity and angular velocity from a Gaussian distribution with mean of optimal values
                mean_velocity_0, mean_angular_velocity_0 = action_0[0], action_0[1]
                mean_velocity_1, mean_angular_velocity_1 = action_1[0], action_1[1]
                sampled_velocity_0 = np.random.normal(mean_velocity_0, 2)
                sampled_angular_velocity_0 = np.random.normal(mean_angular_velocity_0, np.pi)
                sampled_velocity_1 = np.random.normal(mean_velocity_1, 2)
                sampled_angular_velocity_1 = np.random.normal(mean_angular_velocity_1, np.pi)
                
                # Find the action in moves_0 and moves_1 that is nearest to the sampled values
                distances_0 = np.linalg.norm(np.array(moves_0) - [sampled_velocity_0, sampled_angular_velocity_0], axis=1)
                distances_1 = np.linalg.norm(np.array(moves_1) - [sampled_velocity_1, sampled_angular_velocity_1], axis=1)
                nearest_index_0 = np.argmin(distances_0)
                nearest_index_1 = np.argmin(distances_1)
                
                action_0 = moves_0[nearest_index_0]
                action_1 = moves_1[nearest_index_1]
                
                action = action_0 + action_1

        elif current_rollout_node.state.timestep > Game.config.rollout_length:
            # choose the action that turns the agents towards goal direction
            weights_angle_0 = [mm_unicycle(current_rollout_node.state.get_state_0(), action, delta_t=Game.config.delta_t)[2]%(np.pi) for action in moves_0]
            weights_angle_1 = [mm_unicycle(current_rollout_node.state.get_state_1(), action, delta_t=Game.config.delta_t)[2]%(np.pi) for action in moves_1]
            weights_dist_0 = [distance(mm_unicycle(current_rollout_node.state.get_state_0(), action, delta_t=Game.config.delta_t), [Game.config.finish_line, current_rollout_node.state.y0]) for action in moves_0]
            weights_dist_1 = [distance(mm_unicycle(current_rollout_node.state.get_state_1(), action, delta_t=Game.config.delta_t), [Game.config.finish_line, current_rollout_node.state.y1]) for action in moves_1]
            action_0 = moves_0[np.argmin(np.add(weights_angle_0, weights_dist_0))]
            action_1 = moves_1[np.argmin(np.add(weights_angle_1, weights_dist_1))]
            action = action_0 + action_1
        return action
            
    def backpropagate(self, Game, payoff_list):
        # backpropagate statistics of the node
        self._number_of_visits += 1
        
        for agent in Game.Model_params["agents"]:
            self._aggr_payoffs[agent] += float(payoff_list[agent])

        if self.parent:
            if not [self.state.x0, self.state.y0, self.state.theta0, self.state.timestep] in Game.forbidden_states:
                if self.parent.select_policy_data['action_stats_0'].get(str(self.parent_action[:2])): # update parent stats
                    self.parent.select_policy_data['action_stats_0'][str(self.parent_action[:2])]['num_count'] += 1
                    self.parent.select_policy_data['action_stats_0'][str(self.parent_action[:2])]['sum_payoffs'] += float(payoff_list[0])
                    #print("backpropagate agent 0: {}".format(self.parent_action[:2]))
                else: # create stats
                    self.parent.select_policy_data['action_stats_0'][str(self.parent_action[:2])] = {'num_count': 1, 'sum_payoffs': 0}
                    self.parent.select_policy_data['action_stats_0'][str(self.parent_action[:2])]['action'] = self.parent_action[:2]
                    #print("create action stat for agent 0: {}".format(self.parent_action[:2]))
            if not [self.state.x1, self.state.y1, self.state.theta1, self.state.timestep] in Game.forbidden_states:
                if  self.parent.select_policy_data['action_stats_1'].get(str(self.parent_action[2:])): # update parent stats
                    self.parent.select_policy_data['action_stats_1'][str(self.parent_action[2:])]['num_count'] += 1
                    self.parent.select_policy_data['action_stats_1'][str(self.parent_action[2:])]['sum_payoffs'] += float(payoff_list[1])
                    #print("backpropagate agent 1: {}".format(self.parent_action[2:]))
                else: # create stats
                    self.parent.select_policy_data['action_stats_1'][str(self.parent_action[2:])] = {'num_count': 1, 'sum_payoffs': 0}
                    self.parent.select_policy_data['action_stats_1'][str(self.parent_action[2:])]['action'] = self.parent_action[2:]
                    #print("create action stat for agent 1: {}".format(self.parent_action[2:]))
            self.parent.backpropagate(Game, payoff_list)

    def _tree_policy(self, Game):
        current_node = self
        while not is_terminal(Game, current_node.state):
            #print("Tree policy current node: {}".format(current_node.state.get_state_together()))
            #print("Current node not terminal")

            if Game.config.feature_flags['expansion_policy']['every-child']:
                if not current_node.is_fully_expanded():
                    #print("Current node not fully expanded")
                    return current_node.expand(Game)
                else:
                    #print("Current node fully expanded, next child")
                    current_node = current_node.select_policy_child(Game)
                    # prevent parent from choosing the same action again
            elif Game.config.feature_flags['expansion_policy']['random']:
                #randomm expansion
                """random_selector = random.choice([True, False])
                print(len(current_node.children))
                if random_selector or len(current_node.children) <= 50:
                    return current_node.expand(Game)
                else:
                    current_node = current_node.select_policy_child(Game)"""
        #print("Tree policy current node: {}".format(current_node.state.get_state_together()))
        return current_node