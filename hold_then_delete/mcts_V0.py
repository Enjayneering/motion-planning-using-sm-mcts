import numpy as np
import itertools
import logging
from common import timestep_max, env

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
        x_distance = self.x1 - self.x2
        progress_and_lead = 2*self.x1 + x_distance
        distance_to_goal_state = np.exp(-abs(10-self.x1)-abs(5-self.y1))
        return int(progress_and_lead)
    
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
        self._UCT_value = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        # create a list of two dictionaries for each agent to store the regrets of each action
        # self._regrets = None
        # self._regrets = self.build_regret_table()
        # print(self._regrets)
        
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def best_child(self):
        # UCT: returns the child with the highest UCB1 score
        # c_param: exploration parameter
        # TODO: check, if sum of rewards has to be normalized
        choices_weights = [c.calc_UCT(c_param=0.1) for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def robust_child(self):
        choices_weights = [c.n() for c in self.children]
        return self.children[np.argmax(choices_weights)]
    
    def calc_UCT(self, c_param=0.1):
        UCT = (self.X() / self.n()) + c_param * np.sqrt((2 * np.log(self.parent.n()) / self.n()))
        #UCT_1 = (-self.X() / self.n()) + c_param * np.sqrt((2 * np.log(self.parent.n()) / self.n()))
        return UCT
    
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
        return current_rollout_state.get_reward()
    
    def backpropagate(self, utility):
        # backpropagate statistics of the node
        self._number_of_visits += 1
        self._sum_of_rewards_X += utility
        if self.parent:
            self._UCT_value = self.calc_UCT()
            self.parent.backpropagate(utility)
    
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal():
            print("Current node not terminal")
            if not current_node.is_fully_expanded():
                print("Current node not fully expanded")
                return current_node.expand()
            else:
                print("Current node fully expanded, next child")
                current_node = current_node.best_child()
                print("Best child: {}, Utility: {}".format(current_node.state.get_state(), current_node._UCT_value))
        return current_node
    
    def next_node(self):
        simulation_count = 1000

        for _ in range(simulation_count):
            print("Simulation: {}".format(_))
            print("Starting tree policy")
            v = self._tree_policy()
            print("Starting rollout")
            reward = v.rollout()
            print("Backpropagating")
            v.backpropagate(reward)
        self._next_child = self.robust_child()
        return self._next_child