import numpy as np
import copy
import itertools
import logging
from common import timestep_max, env, c_param, a_progress, a_lead

logging.basicConfig(filename='mcts.log', encoding='utf-8', level=logging.DEBUG)

class State:
    def __init__(self, x1, y1, x2, y2, timestep):
        self.x1 = x1
        self.y1 = y1 
        self.x2 = x2
        self.y2 = y2
        self.timestep = timestep
    
    def get_legal_actions(self):
        # returns a list of all possible actions [[x1, y1, x2, y2], ...]
        global env
        values = [-1, 0, 1]
        action_tuples = itertools.product(values, repeat=4)
        sampled_actions = [list(action) for action in action_tuples]
        
        #prune non-valid actions
        sampled_actions_pruned = []
        for action in sampled_actions:
            if env.is_in_free_space(self.x1 + action[0], self.y1 + action[1]) and env.is_in_free_space(self.x2 + action[2], self.y2 + action[3]):
                sampled_actions_pruned.append(action)
        return sampled_actions_pruned
    
    def get_legal_actions_seperate(self):
        # returns a list of possible actions for each agent [[x1, y1], ...], [[x2, y2], ...]
        global env
        values = [-1, 0, 1]
        action_tuples = itertools.product(values, repeat=2)
        sampled_actions_0 = [list(action) for action in action_tuples]
        sampled_actions_1 = copy.deepcopy(sampled_actions_0)
    
        #prune non-valid actions
        sampled_actions_0 = [action for action in sampled_actions_0 if env.is_in_free_space(self.x1 + action[0], self.y1 + action[1])]
        sampled_actions_1 = [action for action in sampled_actions_1 if env.is_in_free_space(self.x2 + action[0], self.y2 + action[1])]

        return sampled_actions_0, sampled_actions_1
    
    def move(self, action):
        # transition-function: get new state from previous state and chosen action
        return State(self.x1 + action[0], self.y1 + action[1], self.x2 + action[2], self.y2 + action[3], self.timestep + 1)

    def is_terminal(self):
        # terminal condition
        if self.timestep >= timestep_max:
            return True
        return False
    
    # Terminal Reward
    def get_reward(self): 
        # TODO: possibly extending with projection onto centerline
        euclidean_distance = np.sqrt((self.x1 - self.x2)**2+(self.y1 - self.y2)**2)
        x_lead = self.x1 - self.x2
        progress_and_lead = a_progress*self.x1 + a_lead*x_lead
        distance_to_goal_state = np.exp(-abs(10-self.x1)-abs(5-self.y1))
        return int(x_lead)
    
    def get_state(self):
        logging.debug("State: {}".format([self.x1, self.y1, self.x2, self.y2, self.timestep]))
        return [self.x1, self.y1, self.x2, self.y2, self.timestep]

class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._next_child = None
        self._number_of_visits = 1
        self._sum_of_rewards_X = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.action_stats_0 = self.init_action_stat_0() # [action, number_count, sum of payoffs]
        self.action_stats_1 = self.init_action_stat_1() # [action, number_count, sum of payoffs]
        # create a list of two dictionaries for each agent to store the regrets of each action
        # self._regrets = None
        # self._regrets = self.build_regret_table()
        # print(self._regrets)
    
    def init_action_stat_0(self):
        legal_action_0, legal_action_1 = self.state.get_legal_actions_seperate()
        self.action_stats_0 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_0]
        return self.action_stats_0
    
    def init_action_stat_1(self):
        legal_action_0, legal_action_1 = self.state.get_legal_actions_seperate()
        self.action_stats_1 = [{"action": action, "num_count": 1, "sum_payoffs": 0} for action in legal_action_1]
        return self.action_stats_1

    def update_action_stats(self):
        for i, action_stat in enumerate(self.action_stats_0):
            self.action_stats_0[i]["num_count"] = 1
            self.action_stats_0[i]["sum_payoffs"] = 0
            for c in self.children:
                if c.parent_action[:2] == action_stat["action"]:
                    self.action_stats_0[i]["num_count"] += 1
                    self.action_stats_0[i]["sum_payoffs"] += c.X()
        
        for i, action_stat in enumerate(self.action_stats_1):
            self.action_stats_1[i]["num_count"] = 1
            self.action_stats_1[i]["sum_payoffs"] = 0
            for c in self.children:
                if c.parent_action[2:] == action_stat["action"]:
                    self.action_stats_1[i]["num_count"] += 1
                    self.action_stats_1[i]["sum_payoffs"] += -c.X()
    
    def calc_UCT(self, action_stat, reward_range, c_param):
            X = action_stat["sum_payoffs"]
            n = action_stat["num_count"]
            UCT = (X/n) + c_param *reward_range* np.sqrt((np.log(self.n()) / n)) # reward range normalizes the payoffs so that we get better exploration
            return UCT
    
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def select_action(self, reward_range):
        # UCT: returns the child with the highest UCB1 score
        # c_param: exploration parameter
        # TODO: check, if sum of rewards has to be normalized
        choices_weights_0 = [self.calc_UCT(action_stat, reward_range, c_param=c_param) for action_stat in self.action_stats_0]
        action_select_0 = self.action_stats_0[np.argmax(choices_weights_0)]["action"]
        
        choices_weights_1 = [self.calc_UCT(action_stat, reward_range, c_param=c_param) for action_stat in self.action_stats_1]
        action_select_1 = self.action_stats_1[np.argmax(choices_weights_1)]["action"]
        selected_action = action_select_0 + action_select_1
        print("Action stats 0: {}".format(self.action_stats_0))
        print("Action stats 1: {}".format(self.action_stats_1))
        print("Selected action: {}".format(selected_action))
        return selected_action
    
    def select_child(self, reward_range):
        selected_action = self.select_action(reward_range)
        best_child = [child for child in self.children if child.parent_action == selected_action]
        return best_child[0]
    
    def robust_child(self):
        choices_weights = [c.n() for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def n(self):
        return self._number_of_visits
    
    def X(self):
        return self._sum_of_rewards_X


    def expand(self):
        action = self._untried_actions.pop() # TODO: which action to pop, maybe random?
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        print("Child node: {}".format(child_node.state.get_state()))
        return child_node
    
    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def rollout(self):
        # rollout policy: random action selection
        current_rollout_state = self.state

        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
            print("Rollout State: {}".format(current_rollout_state.get_state()))
        reward = current_rollout_state.get_reward()
        return reward
    
    def backpropagate(self, reward):
        # backpropagate statistics of the node
        self._number_of_visits += 1
        self._sum_of_rewards_X += reward
        if self.parent:
            self.parent.backpropagate(reward)
    
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self, reward_range=0):
        current_node = self
        while not current_node.is_terminal():
            print("Current node not terminal")
            if not current_node.is_fully_expanded():
                print("Current node not fully expanded")
                return current_node.expand()
            else:
                print("Current node fully expanded, next child")
                current_node = current_node.select_child(reward_range)
        return current_node