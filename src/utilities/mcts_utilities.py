# import modules
import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS')
from utilities.common_utilities import *

def tree_policy(env, curr_node, curr_timestep, max_timestep, config_mcts):
    while not is_terminal(env, curr_node.joint_state, curr_timestep, max_timestep):
        #print("Tree policy current node: {}".format(current_node.state.get_state_together()))
        #print("Current node not terminal")

        if config_mcts['expansion_policy']['every-child']:
            if not current_node.is_fully_expanded():
                # expand one child and return
                return current_node.expand()
            else:
                #print("Current node fully expanded, next child")
                current_node = current_node.select_policy_child(Game)
                # prevent parent from choosing the same action again
        elif Game.config.feature_flags['expansion_policy']['random-informed']:
            moves_0, moves_1, moves_joint = sample_legal_actions(Game, current_node.state)
            action_to_expand = self._sample_action_informed_random(Game, current_node, moves_0, moves_1)
            if any(action_to_expand == sublist for sublist in current_node._tried_actions):
                current_node = current_node.select_policy_child(Game)
            else:
                # expand one child and return
                return current_node.expand(Game, action_to_expand)
    #print("Tree policy current node: {}".format(current_node.state.get_state_together()))
    return current_node

def select_policy(node, config_mcts):
    """
    Select the next child to follow in the tree
    """
    elif config_mcts['selection_policy']['DUCT']:
        return select_DUCT(node)
    elif config_mcts['selection_policy']['EXP3']:
        return select_EXP3(node)
    elif config_mcts['selection_policy']['RM']:
        return select_RM(node)
    else:
        raise ValueError("Invalid selection policy")

######################
# Functions for DUCT #
######################

def select_DUCT(node):
    """Select the best child node based on the UCB1 formula
    """

    pass

######################
# Functions for EXP3 #
######################


####################
# Functions for RM #
####################