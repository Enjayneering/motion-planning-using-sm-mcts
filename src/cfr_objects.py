import sys
import numpy as np
import itertools
import logging
import random
import time

from networkx_utilities import *
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
        state_0 = self.get_state(agent=0)
        action_0 = action[:2]

        state_1 = self.get_state(agent=1)
        action_1 = action[2:]

        x0_new, y0_new, theta0_new = mm_unicycle(state_0, action_0, delta_t=delta_t)
        x1_new, y1_new, theta1_new = mm_unicycle(state_1, action_1, delta_t=delta_t)

        timestep_new = self.timestep + delta_t
        return State(x0_new, y0_new, theta0_new, x1_new, y1_new, theta1_new, timestep_new)

    def get_state(self, agent=0):
        if agent == 0:
            state_list = [self.x0, self.y0, self.theta0]
        elif agent == 1:
            state_list = [self.x1, self.y1, self.theta1]
        return state_list

    def get_state_together(self):
        state_list = [self.x0, self.y0, self.theta0, self.x1, self.y1, self.theta1, self.timestep]
        return state_list
    
#############################
#############################  

class CFRNode:
    def __init__(self, Game, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.actions = {0: sample_legal_actions(Game, self.state)[0], 1: sample_legal_actions(Game, self.state)[1]}

        ########## CFR parameters ##########
        self._num_actions = {0: len(self.actions[0]), 1: len(self.actions[1])}
        self._action_utilities = {0: [0] * self._num_actions[0], 1: [0] * self._num_actions[1]}
        self._regret_sum = {0: [0] * self._num_actions[0], 1: [0] * self._num_actions[1]}
        self._strategy_sum = {0: [0] * self._num_actions[0], 1: [0] * self._num_actions[1]}
        
        self._number_of_visits = 0

        self._untried_actions = self._get_untried_actions(Game)
        self._tried_actions = []
        self._select_policy_stats = {"action_stats_0": {}, "action_stats_1": {}}
        
    def get_strategy(self, agent=0):
        normalizing_sum = sum(max(regret, 0) for regret in self._regret_sum[agent])
        strategy = [max(regret, 0) / normalizing_sum if normalizing_sum > 0 else 1.0 / len(self._regret_sum[agent])
                    for action, regret in enumerate(self._regret_sum[agent])]
        self._strategy_sum[agent] = [self._strategy_sum[agent][action] + strategy[action]
                             for action in range(self._num_actions[agent])]
        return strategy
    
    def update_regret_sum(self, action, regret, agent=0):
        self._regret_sum[agent][action] += max(regret, 0)
        print("Regret sum: {}".format(self._regret_sum))
    
    """def get_average_strategy(self, agent=0):
        avg_strategy = np.zeros(self._num_actions[agent])
        normalizing_sum = 0
        
        for a in range(self._num_actions[agent]):
            normalizing_sum += self._strategy_sum[agent][a]
        for a in range(self._num_actions[agent]):
            if normalizing_sum > 0:
                avg_strategy[a] = self._strategy_sum[agent][a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / self._num_actions[agent]
		
        return avg_strategy"""

    def expand(self, Game, action=None):
        if action is None:
            # Pop random action out of the list
            action = self._untried_actions.pop(np.random.randint(len(self._untried_actions)))
        else:
            # Delete the specified action from the list
            self._untried_actions.remove(action)
        self._tried_actions.append(action)

        next_state = self.state.move(action, delta_t=Game.Model_params["delta_t"])

        child_node = CFRNode(Game, next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        #print("Child node: {}".format(child_node.state.get_state_together()))
        return child_node

    def _get_untried_actions(self, Game):
        untried_action_0, untried_action_1, untried_actions = sample_legal_actions(Game, self.state)
        return untried_actions
    
    def _separate_actions(self, actions_together):
        # Remove duplicate actions
        actions_0 = list(set(tuple(action[:2]) for action in actions_together))
        actions_1 = list(set(tuple(action[2:]) for action in actions_together))
        actions_0 = [list(action) for action in actions_0]
        actions_1 = [list(action) for action in actions_1]
        random.shuffle(actions_0)
        random.shuffle(actions_1)
        return actions_0, actions_1

    # TODO: vary Selection policy
    def calc_UCT(self, action_stat, c_param):
            X = action_stat["sum_payoffs"]
            n = action_stat["num_count"]
            UCT = (X/n) + c_param * np.sqrt((np.log(self.n()) / n)) # payoff range normalizes the payoffs so that we get better exploration
            return UCT
    

    
    def _select_action_UCT(self, Game, agent = 0):
        weights = [self.calc_UCT(action_stat, Game.MCTS_params['c_param']) for action_stat in self._select_policy_stats['action_stats_{}'.format(agent)].values()]
        action_select = list(self._select_policy_stats['action_stats_{}'.format(agent)].values())[np.argmax(weights)]['action']
        return action_select
    
    def _select_action_max(self, Game, agent = 0):
        weights = [action_stat["sum_payoffs"]/action_stat['num_count'] for action_stat in self._select_policy_stats['action_stats_{}'.format(agent)].values()]
        action_select = list(self._select_policy_stats['action_stats_{}'.format(agent)].values())[np.argmax(weights)]['action']
        return action_select
    
    def _select_action_robust(self, Game, agent = 0):

        weights = self._strategy_sum[agent]
        
        print("Actions to choose Agent {}: {}".format(agent, self.actions[agent]))
        print("Weights num count: {}".format(weights))
        
        if Game.config.feature_flags['strategy']['pure']:
            action_select = self.actions[agent][np.argmax(weights)]
            return action_select
        elif Game.config.feature_flags['strategy']['mixed']:
            probabilitic_sample_index = np.random.choice(range(len(weights)), p=np.array(weights)/ np.sum(weights))
            action_select = self.actions[agent][probabilitic_sample_index]
            return action_select
        

    def select_policy_child(self, Game):
        # resolve selecting actions that lead to a collision
        """if Game.config.feature_flags['collision_handling']['pruning']:
            if Game.config.feature_flags['selection_policy']['uct-decoupled']:
                weights_0 = [self.calc_UCT(action_stat, Game.MCTS_params['c_param']) for action_stat in self._select_policy_stats['action_stats_0'].values()]
                weights_1 = [self.calc_UCT(action_stat, Game.MCTS_params['c_param']) for action_stat in self._select_policy_stats['action_stats_1'].values()]
            elif Game.config.feature_flags['selection_policy']['greedy']:
                weights_0 = [action_stat["sum_payoffs"] for action_stat in self._select_policy_stats['action_stats_0'].values()]
                weights_1 = [action_stat["sum_payoffs"] for action_stat in self._select_policy_stats['action_stats_1'].values()]
            while True:
                selected_action_0 = list(self._select_policy_stats['action_stats_0'].values())[np.argmax(weights_0)]['action']
                selected_action_1 = list(self._select_policy_stats['action_stats_1'].values())[np.argmax(weights_1)]['action']
                x0, y0, theta0 = mm_unicycle(self.state.get_state(agent=0), selected_action_0, delta_t=Game.Model_params["delta_t"])
                x1, y1, theta1 = mm_unicycle(self.state.get_state(agent=1), selected_action_1, delta_t=Game.Model_params["delta_t"])
                if distance([x0, y0], [x1, y1]) < Game.Model_params["collision_distance"]:
                    weights_0[np.argmax(weights_0)] = -np.inf
                    weights_1[np.argmax(weights_1)] = -np.inf
                    #print("Weights_0: {}, Weights_1: {}".format(weights_0, weights_1))
                    #print("Collision detected from state {} to state {}".format(self.state.get_state_together(), [x0, y0, theta0, x1, y1, theta1, self.state.timestep+Game.Model_params["delta_t"]]))
                else:
                    break
        elif Game.config.feature_flags['collision_handling']['punishing']:
            if Game.config.feature_flags['selection_policy']['uct-decoupled']:
                selected_action_0 = self._select_action_UCT(Game, agent=0)
                selected_action_1 = self._select_action_UCT(Game, agent=1)
            elif Game.config.feature_flags['selection_policy']['max']:
                selected_action_0 = self._select_action_max(Game, agent=0)
                selected_action_1 = self._select_action_max(Game, agent=1)

        selected_action = selected_action_0 + selected_action_1
        #print( "Selected action: {}".format(selected_action))
        if any(selected_action == sublist for sublist in self._tried_actions):
            child = [child for child in self.children if child.parent_action == selected_action][0]
        else:
            #expand tree
            child = self.expand(Game, selected_action)
        #print("Selected child: {}".format(child.state.get_state_together()))"""

        strategy_0 = self.get_strategy(agent=0)
        strategy_1 = self.get_strategy(agent=1)
        action_0_ix =  np.random.choice(range(len(strategy_0)), p=np.array(strategy_0)/ np.sum(strategy_0))
        action_1_ix = np.random.choice(range(len(strategy_1)), p=np.array(strategy_1)/ np.sum(strategy_1))
        select_action_0 = self.actions[0][action_0_ix]
        select_action_1 = self.actions[1][action_1_ix]
        selected_action = select_action_0 + select_action_1
        if any(selected_action == sublist for sublist in self._tried_actions):
            child = [child for child in self.children if child.parent_action == selected_action][0]
        else:
            #expand tree
            child = self.expand(Game, selected_action)
        return child

    def _robust_child_joint(self):
        choices_weights = [c.n() for c in self.children]
        index = np.argmax(choices_weights)
        robust_child = self.children[index]
        return robust_child
    
    
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
        elif Game.config.feature_flags['final_move']['max']:
            selected_action_0 = self._select_action_max(Game, agent=0)
            selected_action_1 = self._select_action_max(Game, agent=1)
            selected_action = selected_action_0 + selected_action_1
            child = [child for child in self.children if child.parent_action == selected_action][0]
            print("Best child: {}".format(child.state.get_state_together()))
        #print("Expected Payoff Value Agent 0 at timestep {}: {}".format(self.state.timestep, child.parent._select_policy_stats['action_stats_0'][str(child.parent_action[:2])]['sum_payoffs']/child.parent._select_policy_stats['action_stats_0'][str(child.parent_action[:2])]['num_count']))
        #print("Expected Payoff Value Agent 1 at timestep {}: {}".format(self.state.timestep, child.parent._select_policy_stats['action_stats_1'][str(child.parent_action[2:])]['sum_payoffs']/child.parent._select_policy_stats['action_stats_1'][str(child.parent_action[2:])]['num_count']))
        #print("Selection Probability Agent 0: {}".format(child.parent._select_policy_stats['action_stats_0'][str(child.parent_action[:2])]['num_count']/child.parent._number_of_visits))
        #print("Selection Probability Agent 1: {}".format(child.parent._select_policy_stats['action_stats_1'][str(child.parent_action[2:])]['num_count']/child.parent._number_of_visits))
        return child

    def n(self):
        return self._number_of_visits

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def rollout(self, Game, max_timestep=None):
        # rollout policy: random action selection
        current_rollout_node = self

        rollout_trajectory = [current_rollout_node.state]
        
        interm_payoff_list_sum = []
        final_payoff_list_sum = []

        # intermediate timesteps
        while not is_terminal(Game, current_rollout_node.state, max_timestep=max_timestep):
            #print("Rollout State: {}".format(current_rollout_node.state.get_state_together()))
            moves_0, moves_1, possible_moves = sample_legal_actions(Game, current_rollout_node.state)

            #print("Moves 0: {}, Moves 1: {}".format(moves_0, moves_1))
            #print("Possible moves: {}".format(possible_moves))

            #TODO: change parameter punishment when stuck (or pruning..?)

            if len(moves_0) == 0 and len(moves_1) == 0:
                Game.forbidden_states.append(current_rollout_node.state.get_state(agent=0)+[current_rollout_node.state.timestep])
                Game.forbidden_states.append(current_rollout_node.state.get_state(agent=1)+[current_rollout_node.state.timestep])
                print("Both agents stuck in environment, append current state on forbidden list, break")
                break
            elif len(moves_0) == 0:
                Game.forbidden_states.append(current_rollout_node.state.get_state(agent=0)+[current_rollout_node.state.timestep])
                print("Agent 0 stuck in environment, append current state on forbidden list, break")
                break
            elif len(moves_1) == 0:
                Game.forbidden_states.append(current_rollout_node.state.get_state(agent=1)+[current_rollout_node.state.timestep])
                print("Agent 1 stuck in environment, append current state on forbidden list, break")
                break

            #print("Forbidden states: {}".format(Game.forbidden_states))

            # choose action due to rollout policy
            action = self.rollout_policy(Game, current_rollout_node, moves_0, moves_1, possible_moves)

            #print("Rollout Action: {}".format(action))
            next_rollout_state = current_rollout_node.state.move(action, delta_t=Game.Model_params["delta_t"])
            next_rollout_node = CFRNode(Game, next_rollout_state, parent=current_rollout_node, parent_action=action)

            # updating intermediate payoffs
            intermediate_payoff_sum_incr, intermediate_payoff_each_incr = get_intermediate_payoffs(Game, current_rollout_node.state, next_rollout_node.state, discount_factor=Game.config.discount_factor)
            interm_payoff_list_sum.append(intermediate_payoff_sum_incr)

            current_rollout_node = next_rollout_node
            rollout_trajectory.append(current_rollout_node.state)

        # updating final payoffs
            
        #if agent_has_finished(Game, current_rollout_node.state, agent=0) or agent_has_finished(Game, current_rollout_node.state, agent=1):
        #    final_payoff_sum_incr, final_payoff_each_incr = get_final_payoffs(Game, current_rollout_node.state)
        #    final_payoff_list_sum.append(final_payoff_sum_incr)
        if is_terminal(Game, current_rollout_node.state, max_timestep=max_timestep):
            final_payoff_sum_incr, final_payoff_each_incr = get_final_payoffs(Game, current_rollout_node.state)
            final_payoff_list_sum.append(final_payoff_sum_incr)

        accumulated_payoff_list = get_total_payoffs(Game, interm_payoff_list_sum, final_payoff_list_sum)

        return rollout_trajectory, accumulated_payoff_list
    
    def _get_action_heuristic(self, Game, current_node, moves_0, moves_1):
        # get closest terminal state and the angle to it
        if Game.env.finish_line is not None:
            terminal_state_0 = [Game.env.finish_line, current_node.state.y0] # [x,y]
            terminal_state_1 = [Game.env.finish_line, current_node.state.y1]
        elif Game.env.goal_state is not None:
            terminal_state_0 = [Game.env.goal_state['x0'], Game.env.goal_state['y0']]
            terminal_state_1 = [Game.env.goal_state['x1'], Game.env.goal_state['y1']]
        angle_to_goal_0 = angle_to_goal(current_node.state.get_state(agent=0), terminal_state_0)
        angle_to_goal_1 = angle_to_goal(current_node.state.get_state(agent=1), terminal_state_1)

        # best reference action (heuristic)
        weights_angle_0 = []
        for action in moves_0:
            prev_state = current_node.state.get_state(agent=0)
            new_state = mm_unicycle(prev_state, action, delta_t=Game.config.delta_t)
            new_theta = new_state[2]
            new_theta_circle = np.fmod(new_theta, 2 * np.pi)
            result = new_theta_circle - angle_to_goal_0
            weights_angle_0.append(np.abs(result))
        weights_angle_0 = [np.abs(np.fmod(mm_unicycle(current_node.state.get_state(agent=0), action, delta_t=Game.config.delta_t)[2],(2*np.pi))-angle_to_goal_0) for action in moves_0]
        weights_angle_1 = [np.abs(np.fmod(mm_unicycle(current_node.state.get_state(agent=1), action, delta_t=Game.config.delta_t)[2],(2*np.pi))-angle_to_goal_1) for action in moves_1]
        weights_dist_0 = [distance(mm_unicycle(current_node.state.get_state(agent=0), action, delta_t=Game.config.delta_t), terminal_state_0) for action in moves_0]
        weights_dist_1 = [distance(mm_unicycle(current_node.state.get_state(agent=1), action, delta_t=Game.config.delta_t), terminal_state_1) for action in moves_1]

        # scale weights so that we get best trade of of getting close and turning towards goal
        epsilon = 1e-7  # Small constant to prevent division by zero

        # Normalize weights_angle and weights_dist
        weights_angle_norm_0 = weights_angle_0 / (np.max(weights_angle_0) + epsilon)
        weights_dist_norm_0 = weights_dist_0 / (np.max(weights_dist_0) + epsilon)

        weights_angle_norm_1 = weights_angle_1 / (np.max(weights_angle_1) + epsilon)
        weights_dist_norm_1 = weights_dist_1 / (np.max(weights_dist_1) + epsilon)

        # Combine the normalized weights
        weights_0 = weights_angle_norm_0 + weights_dist_norm_0
        weights_1 = weights_angle_norm_1 + weights_dist_norm_1

        action_0 = moves_0[np.argmin(weights_0)]
        action_1 = moves_1[np.argmin(weights_1)]
        return action_0, action_1
    
    def _sample_action_informed_random(self, Game, current_node, moves_0, moves_1):
        # best reference action (heuristic)
        action_0, action_1 = self._get_action_heuristic(Game, current_node, moves_0, moves_1)
        
        # Sample velocity and angular velocity from a Gaussian distribution with mean of optimal values
        mean_velocity_0, mean_angular_velocity_0 = action_0[0], action_0[1]
        mean_velocity_1, mean_angular_velocity_1 = action_1[0], action_1[1]

        sampled_velocity_0 = np.random.normal(mean_velocity_0, Game.config.standard_dev_vel_0)
        sampled_angular_velocity_0 = np.random.normal(mean_angular_velocity_0, Game.config.standard_dev_ang_vel_0)
        sampled_velocity_1 = np.random.normal(mean_velocity_1, Game.config.standard_dev_vel_1)
        sampled_angular_velocity_1 = np.random.normal(mean_angular_velocity_1, Game.config.standard_dev_ang_vel_1)
        
        # Find the action in moves_0 and moves_1 that is nearest to the sampled values
        distances_0 = np.linalg.norm(np.array(moves_0) - [sampled_velocity_0, sampled_angular_velocity_0], axis=1)
        distances_1 = np.linalg.norm(np.array(moves_1) - [sampled_velocity_1, sampled_angular_velocity_1], axis=1)
        nearest_index_0 = np.argmin(distances_0)
        nearest_index_1 = np.argmin(distances_1)

        action_0 = moves_0[nearest_index_0]
        action_1 = moves_1[nearest_index_1]
        
        action = action_0 + action_1
        return action

    def rollout_policy(self, Game, current_node, moves_0, moves_1, possible_moves):
        if current_node.state.timestep <= Game.config.max_timehorizon*Game.config.alpha_rollout:
            if Game.config.feature_flags['rollout_policy']['random-uniform']:
                # choose a random action to simulate rollout
                action = possible_moves[np.random.randint(len(possible_moves))]
            elif Game.config.feature_flags['rollout_policy']['random-informed']:
                action = self._sample_action_informed_random(Game, current_node, moves_0, moves_1)
        else:
            # choose the action that turns the agents towards goal direction
            action_0, action_1 = self._get_action_heuristic(Game, current_node, moves_0, moves_1)
            action = action_0 + action_1
        return action
            
    def backpropagate(self, Game, payoff_list, fixed_action=None):
        # backpropagate statistics of the node
        self._number_of_visits += 1

        # backpropagate if there is a parent
        if not [self.state.x0, self.state.y0, self.state.theta0, self.state.timestep] in Game.forbidden_states:
            for a in range(self._num_actions[0]):
                regret = payoff_list[0] - max(self._action_utilities[0][a], 0) if self.actions[0][a] == fixed_action[:2]  else 0-max(self._action_utilities[0][a], 0)
                self.update_regret_sum(a, regret, agent=0)
            self._action_utilities[0][self.actions[0].index(fixed_action[:2])] += payoff_list[0]
        if not [self.state.x1, self.state.y1, self.state.theta1, self.state.timestep] in Game.forbidden_states:
            for a in range(self._num_actions[1]):
                regret = payoff_list[1] - max(self._action_utilities[1][a], 0) if self.actions[1][a] == fixed_action[2:] else 0-max(self._action_utilities[1][a], 0)
                self.update_regret_sum(a, regret, agent=1)
            self._action_utilities[1][self.actions[1].index(fixed_action[2:])] += payoff_list[1]
        
        if self.parent is not None:
            self.parent.backpropagate(Game, payoff_list, fixed_action = self.parent_action)
        else:
            pass

    def _tree_policy(self, Game, max_timestep=None):
        current_node = self
        while not is_terminal(Game, current_node.state, max_timestep = max_timestep):

            if Game.config.feature_flags['expansion_policy']['every-child']:
                if not current_node.is_fully_expanded():
                    # expand one child and return
                    return current_node.expand(Game)
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
    
def run_cfr_mcts(Game, root_state, max_timestep=None):
    start_time = time.time()

    # create Node for local tree search
    init_state = State(x0=root_state.x0,
                        y0=root_state.y0,
                        theta0=root_state.theta0,
                        x1=root_state.x1,
                        y1=root_state.y1,
                        theta1=root_state.theta1,
                        timestep=0)
    current_node = CFRNode(Game=Game, state=init_state)
    
    # ONE STEP OF THE GAME
    for iter in range(Game.MCTS_params['num_iter']):
        #print("Horizon {} | Iteration {}".format(root_state.timestep, iter))
        walking_node = current_node._tree_policy(Game, max_timestep=max_timestep)

        #Simulation / Rollout
        rollout_traj_accum = []
        payoff_accum = [0]*len(Game.Model_params["agents"])
        for k in range(Game.config.k_samples):
            rollout_trajectory, rollout_payoff_list = walking_node.rollout(Game, max_timestep=max_timestep)
            rollout_traj_accum.append(rollout_trajectory)
            payoff_accum = [x + y for x, y in zip(payoff_accum, rollout_payoff_list)]


        # write every x rollout trajectories
        if Game.config.feature_flags["run_mode"]["test"]:
            if iter % freq_stat_data == 0:
                csv_write_rollout_last(Game, rollout_trajectory, timehorizon = current_node.state.timestep, config=Game.config)
       
        # BACKPROPAGATION                
        walking_node.parent.backpropagate(Game, rollout_payoff_list, fixed_action=walking_node.parent_action)

    next_node = current_node.select_final_child(Game=Game)

    if Game.config.feature_flags["run_mode"]["test"]:
        save_tree_to_file(current_node, path_to_tree.format(root_state.timestep))
    
    runtime = time.time() - start_time 

    # transfer search node to global state
    next_global_state = State(x0=next_node.state.x0,
                            y0=next_node.state.y0,
                            theta0=next_node.state.theta0,
                            x1=next_node.state.x1,
                            y1=next_node.state.y1,
                            theta1=next_node.state.theta1,
                            timestep=root_state.timestep+1)
    return next_global_state, runtime
