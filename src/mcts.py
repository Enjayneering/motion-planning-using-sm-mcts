import numpy as np
import itertools
import logging

from common import *
from csv_utilities import *
from payoff_utilities import *
from kinodynamic import *

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

    def move(self, action):
        # transition-function: get new state from previous state and chosen action
        state_0 = self.get_state_0()
        action_0 = action[:2]

        state_1 = self.get_state_1()
        action_1 = action[2:]

        x0_new, y0_new, theta0_new = mm_unicycle(state_0, action_0)
        x1_new, y1_new, theta1_new = mm_unicycle(state_1, action_1)

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
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        ########## MCTS parameters ##########
        self.aggr_payoff_vector = np.ones((Model_params["len_payoff_vector"],1)) # [p_coll_0, p_lead_0, p_coll_1, p_lead_1].T
        #self.aggr_max_payoff_vector = np.zeros(Model_params["len_payoff_vector"],1) # [p_coll_0, p_lead_0, p_coll_1, p_lead_1].T
        
        self._untried_actions = self.untried_actions()
        self.action_stats_0, self.action_stats_1 = self.init_action_stats() # [action, number_count, sum of payoffs]
        self._number_of_visits = 1

    def init_action_stats(self):
        legal_action_0, legal_action_1, _ = sample_legal_actions(self.state)
        self.action_stats_0 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_0]
        self.action_stats_1 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_1]
        return self.action_stats_0, self.action_stats_1

    """def update_max_payoff_vector(self, max_payoff_vector):
        for i, payoff in enumerate(max_payoff_vector):
            if payoff > self.aggr_max_payoff_vector[i]:
                self.aggr_max_payoff_vector[i] = payoff
        pass"""

    def update_action_stats(self):
        # count at each node the number of children visited for the respective action and update that action's stats
        for i, action_stat in enumerate(self.action_stats_0):
            self.action_stats_0[i]["num_count"] = 1
            self.action_stats_0[i]["sum_payoffs"] = 0

            for c in self.children:
                if c.parent_action[:2] == action_stat["action"] and action_stat["action"] != -np.inf:
                    self.action_stats_0[i]["num_count"] += 1
                    self.action_stats_0[i]["sum_payoffs"] += c.X(agent=0)

        for i, action_stat in enumerate(self.action_stats_1):
            self.action_stats_1[i]["num_count"] = 1
            self.action_stats_1[i]["sum_payoffs"] = 0

            for c in self.children:
                if c.parent_action[2:] == action_stat["action"] and action_stat["action"] != -np.inf:
                    self.action_stats_1[i]["num_count"] += 1
                    self.action_stats_1[i]["sum_payoffs"] += c.X(agent=1)
        pass

    def expand(self):
        action = self._untried_actions.pop(np.random.randint(len(self._untried_actions))) #pop random action out of the list

        next_state = self.state.move(action)

        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        #print("Child node: {}".format(child_node.state.get_state_together()))
        return child_node

    def untried_actions(self):
        _, _, _untried_actions = sample_legal_actions(self.state)
        return _untried_actions

    def calc_UCT(self, action_stat, payoff_range, c_param):
            X = action_stat["sum_payoffs"]
            n = action_stat["num_count"]
            UCT = (X/n) + c_param *payoff_range* np.sqrt((np.log(self.n()) / n)) # payoff range normalizes the payoffs so that we get better exploration
            return UCT

    def select_action(self, payoff_range):
        # update action stats based on all possible and already visited childs
        #TODO: action stat update here?
        #print("Try to select action")
        self.update_action_stats()

        # UCT: returns the child with the highest UCB1 score
        # Note: we need to choose the most promising action, but ensuring that they are also collisionfree
        weights_0 = [self.calc_UCT(action_stat, payoff_range, MCTS_params['c_param']) for action_stat in self.action_stats_0]
        weights_1 = [self.calc_UCT(action_stat, payoff_range, MCTS_params['c_param']) for action_stat in self.action_stats_1]

        action_select_0 = self.action_stats_0[np.argmax(weights_0)]["action"]
        action_select_1 = self.action_stats_1[np.argmax(weights_1)]["action"]
        selected_action = action_select_0 + action_select_1
        #print("Weights 0: {}\nWeights 1: {}".format(weights_0, weights_1))
        return selected_action

    def select_child(self, payoff_range):
        # selects action that is itself a Nash Equilibrium
        selected_action = self.select_action(payoff_range)

        if selected_action:
            best_child = [child for child in self.children if child.parent_action == selected_action]
            #print("Selected Child: {}".format(best_child[0].state.get_state_together()))
            return best_child[0]
        else:
            #print("Return self")
            return self  # Return node to choose another joint action


    def robust_child(self):
        choices_weights = [c.n() for c in self.children]
        robust_child = self.children[np.argmax(choices_weights)]
        return robust_child

    #TODO: def optimal_child(self, payoff_range):

    def n(self):
        return self._number_of_visits

    def X(self, agent=None):
        # get sum of payoffs for agent index
        p_coll = self.aggr_payoff_vector[agent,0]
        p_lead = self.aggr_payoff_vector[2+agent,0]
        sum_payoffs = p_coll + p_lead
        return sum_payoffs

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def rollout(self, payoff_weights):
        # rollout policy: random action selection
        current_rollout_node = self

        rollout_trajectory = [current_rollout_node.state]
        
        payoff_vector = np.zeros((Model_params["len_payoff_vector"],1))

        # intermediate timesteps
        while not is_terminal(current_rollout_node.state):
            #print("Rollout State: {}".format(current_rollout_node.state.get_state_together()))
            moves_0, moves_1, possible_moves = sample_legal_actions(current_rollout_node.state)

            #print("Moves 0: {}, Moves 1: {}".format(moves_0, moves_1))
            #print("Possible moves: {}".format(possible_moves))

            #TODO: change parameter punishment when stuck (or pruning..?)
            if len(possible_moves) == 0:
                print("No possible moves")

            if len(moves_0) == 0 and len(moves_1) == 0:
                #payoff_0 += MCTS_params['penalty_stuck_in_env']
                #payoff_1 += MCTS_params['penalty_stuck_in_env']
                print("Both agents stuck in environment, break")
                break
            elif len(moves_0) == 0:
                #payoff_0 += MCTS_params['penalty_stuck_in_env']
                print("Agent 0 stuck in environment, break")
                break
            elif len(moves_1) == 0:
                #payoff_1 += MCTS_params['penalty_stuck_in_env']
                print("Agent 1 stuck in environment, break")
                break

            # choose action due to rollout policy
            action = self.rollout_policy(possible_moves)

            #print("Rollout Action: {}".format(action))
            next_rollout_state = current_rollout_node.state.move(action)
            next_rollout_node = MCTSNode(next_rollout_state, parent=current_rollout_node, parent_action=action)

            # updating intermediate payoffs
            collision_penalty_0, collision_penalty_1 = get_intermediate_penalty(next_rollout_state, payoff_weights)
            #TODO: Multi agent adjustment
            payoff_vector[0] += collision_penalty_0
            payoff_vector[1] += collision_penalty_1

            current_rollout_node = next_rollout_node
            rollout_trajectory.append(current_rollout_node.state)

        # updating final payoffs
        if is_terminal(current_rollout_node.state):
            payoff_final_0, payoff_final_1 = get_final_payoffs(current_rollout_node.state)
            #TODO: Multi agent adjustment
            payoff_vector[2] += payoff_final_0
            payoff_vector[3] += payoff_final_1

        return rollout_trajectory, payoff_vector
    
    def rollout_policy(self, possible_moves):
            #TODO: Check informed rollout sampling
            # choose a random action to simulate rollout
            try:
                action = possible_moves[np.random.randint(len(possible_moves))]
                return action
            except:
                return None
            
    def backpropagate(self, payoff_vector):
        # backpropagate statistics of the node
        self._number_of_visits += 1
        self.aggr_payoff_vector += payoff_vector

        if self.parent:
            self.parent.backpropagate(payoff_vector)

    def _tree_policy(self, payoff_range):
        current_node = self
        while not is_terminal(current_node.state):
            #print("Tree policy current node: {}".format(current_node.state.get_state_together()))
            #print("Current node not terminal")
            if not current_node.is_fully_expanded():
                #print("Current node not fully expanded")
                return current_node.expand()
            else:
                #print("Current node fully expanded, next child")
                current_node = current_node.select_child(payoff_range)
                # prevent parent from choosing the same action again
        #print("Tree policy current node: {}".format(current_node.state.get_state_together()))
        return current_node