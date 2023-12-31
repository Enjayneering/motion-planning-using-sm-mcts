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
        self._number_of_visits = 1
        self._sum_of_payoffs_0 = 0
        self._sum_of_payoffs_1 = 0
        self._untried_actions = self.untried_actions()
        self.action_stats_0, self.action_stats_1 = self.init_action_stats() # [action, number_count, sum of payoffs]

    def init_action_stats(self):
        legal_action_0, legal_action_1, _ = sample_legal_actions(self.state)
        self.action_stats_0 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_0]
        self.action_stats_1 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_1]
        return self.action_stats_0, self.action_stats_1

    def update_action_stats(self):
        # count at each node the number of children visited for the respective action and update that action's stats
        for i, action_stat in enumerate(self.action_stats_0):
            self.action_stats_0[i]["num_count"] = 1
            self.action_stats_0[i]["sum_payoffs"] = 0

            for c in self.children:
                if c.parent_action[:2] == action_stat["action"] and action_stat["action"] != -np.inf:
                    self.action_stats_0[i]["num_count"] += 1
                    self.action_stats_0[i]["sum_payoffs"] += c.X_0()

        for i, action_stat in enumerate(self.action_stats_1):
            self.action_stats_1[i]["num_count"] = 1
            self.action_stats_1[i]["sum_payoffs"] = 0

            for c in self.children:
                if c.parent_action[2:] == action_stat["action"] and action_stat["action"] != -np.inf:
                    self.action_stats_1[i]["num_count"] += 1
                    self.action_stats_1[i]["sum_payoffs"] += c.X_1()

    def expand(self, index=None):
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
        print("Try to select action")
        self.update_action_stats()

        # UCT: returns the child with the highest UCB1 score
        # Note: we need to choose the most promising action, but ensuring that they are also collisionfree
        weights_0 = [self.calc_UCT(action_stat, payoff_range, MCTS_params['c_param']) for action_stat in self.action_stats_0]
        weights_1 = [self.calc_UCT(action_stat, payoff_range, MCTS_params['c_param']) for action_stat in self.action_stats_1]

        action_select_0 = self.action_stats_0[np.argmax(weights_0)]["action"]
        action_select_1 = self.action_stats_1[np.argmax(weights_1)]["action"]
        selected_action = action_select_0 + action_select_1
        print("Weights 0: {}\nWeights 1: {}".format(weights_0, weights_1))

        # resolving immediate collisions by using risk parameter modelling the agents behaviour
        """while is_collision(self.state, selected_action):
            agent_being_considerate = generate_bernoulli(Competitive_params['risk_factor_0'])
            if agent_being_considerate == 0:
                weights_0[np.argmax(weights_0)] = -np.inf
            elif agent_being_considerate == 1:
                weights_1[np.argmax(weights_1)] = -np.inf

            print("Weights 0: {}\nWeights 1: {}".format(weights_0, weights_1))

            action_select_0 = self.action_stats_0[np.argmax(weights_0)]["action"]
            action_select_1 = self.action_stats_1[np.argmax(weights_1)]["action"]
            selected_action = action_select_0 + action_select_1
            print("Selected action: {}".format(selected_action))

            if weights_0[np.argmax(weights_0)] == -np.inf:
                print("Agent 0 stuck in environment")
                if self.parent:
                    action_0_stuck = self.parent_action[:2]
                    # changing value of parent action stat of the action that led to the stuck state
                    for action_stat in self.parent.action_stats_0:
                        if np.array_equal(action_stat["action"], action_0_stuck):
                            # Modify the sum_payoffs field
                            action_stat["sum_payoffs"] = -np.inf  # prevent action from being chosen next time
                    selected_action = None
                break
            if weights_1[np.argmax(weights_1)] == -np.inf:
                print("Agent 1 stuck in environment")
                if self.parent:
                    action_1_stuck = self.parent_action[2:]
                    # changing value of parent action stat of the action that led to the stuck state
                    for action_stat in self.parent.action_stats_1:
                        if np.array_equal(action_stat["action"], action_1_stuck):
                            # Modify the sum_payoffs field
                            action_stat["sum_payoffs"] = -np.inf # prevent action from being chosen next time
                selected_action = None
                break"""
        return selected_action


        #print("Action stats 0: {}".format(self.action_stats_0))
        #print("Action stats 1: {}".format(self.action_stats_1))
        #print("Selected action: {}".format(selected_action))

    def select_child(self, payoff_range):
        # selects action that is itself a Nash Equilibrium
        selected_action = self.select_action(payoff_range)

        if selected_action:
            best_child = [child for child in self.children if child.parent_action == selected_action]
            print("Selected Child: {}".format(best_child[0].state.get_state_together()))
            return best_child[0]
        else:
            print("Return self")
            return self  # Return node to choose another joint action


    def robust_child(self):
        choices_weights = [c.n() for c in self.children]
        robust_child = self.children[np.argmax(choices_weights)]
        return robust_child

    #def optimal_child(self, payoff_range):

    def n(self):
        return self._number_of_visits

    def X_0(self):
        return self._sum_of_payoffs_0

    def X_1(self):
        return self._sum_of_payoffs_1

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def rollout(self, collision_weight):
        # rollout policy: random action selection
        current_rollout_node = self

        rollout_trajectory = [current_rollout_node.state]
        
        payoff_0, payoff_1 = 0, 0
        intermediate_penalties, final_payoff = 0, 0

        # intermediate timesteps
        while not is_terminal(current_rollout_node.state):
            #print("Rollout State: {}".format(current_rollout_node.state.get_state_together()))

            moves_0, moves_1, possible_moves = sample_legal_actions(current_rollout_node.state)

            #print("Moves 0: {}, Moves 1: {}".format(moves_0, moves_1))
            #print("Possible moves: {}".format(possible_moves))

            if len(possible_moves) == 0:
                print("No possible moves")

            if len(moves_0) == 0 and len(moves_1) == 0:
                payoff_0 += MCTS_params['penalty_stuck_in_env']
                payoff_1 += MCTS_params['penalty_stuck_in_env']
                print("Both agents stuck in environment, break")
                break
            elif len(moves_0) == 0:
                payoff_0 += MCTS_params['penalty_stuck_in_env']
                print("Agent 0 stuck in environment, break")
                break
            elif len(moves_1) == 0:
                payoff_1 += MCTS_params['penalty_stuck_in_env']
                print("Agent 1 stuck in environment, break")
                break

            # choose random action
            action = self.rollout_policy(possible_moves)

            #print("Rollout Action: {}".format(action))
            next_rollout_state = current_rollout_node.state.move(action)
            next_rollout_node = MCTSNode(next_rollout_state, parent=current_rollout_node, parent_action=action)

            # updating intermediate payoffs
            collision_penalty_0, collision_penalty_1 = get_collision_penalty(next_rollout_state, collision_weight)
            
            payoff_0 += collision_penalty_0
            payoff_1 += collision_penalty_1

            intermediate_penalties += np.abs(max(collision_penalty_0, collision_penalty_1)) # should be equal for both agents

            current_rollout_node = next_rollout_node
            rollout_trajectory.append(current_rollout_node.state)

        # updating final payoffs
        if is_terminal(current_rollout_node.state):
            payoff_final_0, payoff_final_1 = get_final_payoffs(current_rollout_node.state)
            payoff_0 += payoff_final_0
            payoff_1 += payoff_final_1
            final_payoff += np.abs(max(payoff_final_0, payoff_final_1))

        print(intermediate_penalties, final_payoff)
        return rollout_trajectory, payoff_0, payoff_1, intermediate_penalties, final_payoff
    
    def rollout_policy(self, possible_moves):
            # choose a random action to simulate rollout
            try:
                action = possible_moves[np.random.randint(len(possible_moves))]
                return action
            except:
                return None
            
    def backpropagate(self, payoff_0, payoff_1):
        # backpropagate statistics of the node
        self._number_of_visits += 1
        self._sum_of_payoffs_0 += payoff_0
        self._sum_of_payoffs_1 += payoff_1

        #TODO: check if here or somewhere else
        #self.update_action_stats()
        if self.parent:
            self.parent.backpropagate(payoff_0, payoff_1)

    def _tree_policy(self, payoff_range):
        current_node = self
        while not is_terminal(current_node.state):
            print("Tree policy current node: {}".format(current_node.state.get_state_together()))
            print("Current node not terminal")
            if not current_node.is_fully_expanded():
                print("Current node not fully expanded")
                return current_node.expand()
            else:
                #print("Current node fully expanded, next child")
                current_node = current_node.select_child(payoff_range)
                # prevent parent from choosing the same action again
        print("Tree policy current node: {}".format(current_node.state.get_state_together()))
        return current_node