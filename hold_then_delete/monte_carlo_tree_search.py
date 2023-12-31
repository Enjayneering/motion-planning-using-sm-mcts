import torch
import math
import numpy as np
from game import RaceGame

def score(parent, child):
    """
    The score for an action that would transition between the parent and child.
    """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    if child.visit_count > 0:
        # The value of the child is from the perspective of the opposing player
        value_score = -child.value()
    else:
        value_score = 0
    return

def sample_actions(self, env):
    # TODO: implement collision check and sample only valid actions

    # Generate all possible combinations
    values = [-1, 0, 1]
    sampled_actions = list(np.array(itertools.product(values, repeat=4)))
    return sampled_actions


class Node:
    def __init__(self):
        self.visit_count = 0
        self.value_sum = 0
        self.children = {}
        self.state = None

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0 
        return self.value_sum / self.visit_count #TODO: Check if this is correct

    def is_terminal(self, max_iter):
        pass

    def select_action(self, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        """
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = [action for action in self.children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def select_child(self):
        """
        Select the child with the highest score.
        """
        best_score = -np.inf
        best_action = np.array([0, 0, 0, 0])
        best_child = None

        for action, child in self.children.items():
            score = score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def update(self, action, value): # TODO: Code this
        """
        Update node values from leaf evaluation.
        """
        

    def expand(self, state, action_probs):
        """
        We expand a node
        """
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    # def __repr__(self):
    #     """
    #     Debugger pretty print node info
    #     """
    #     prior = "{0:.2f}".format(self.prior)
    #     return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())


class MCTS:

    def __init__(self, game, args):
        self.game = game
        self.args = args

    def run(self):
        # Initialize root node
        current_node = Node(None)
        # store all visited nodes
        trajectory = [current_node]
        policy = [None]

        # Run main loop
        for _ in range(self.args['num_iters']):
            # return u-value of goal state 
            if len(policy) >= self.args['horizon_goal']:
                break
            node = root
            search_path = [node]


            self.backpropagate(search_path, value, parent.to_play * -1)

        return root

    def backpropagate(self, search_path, value, to_play):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1