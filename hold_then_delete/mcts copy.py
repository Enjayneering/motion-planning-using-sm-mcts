import numpy as np
import random
import copy
import itertools
import pandas as pd
import matplotlib.pyplot as plt
#from game import RaceGame

class Environment:
    # define 2D environment with continuous free space and discretized obstacle space 
    def __init__(self):
        self.x_min = 0
        self.y_min = 0
        self.occupancy_grid = None
        self.occupancy_grid = self.get_occupancy_grid()
        self.x_max = self.get_x_max()
        self.y_max = self.get_y_max()
    
    def get_occupancy_grid(self):
        occupancy_grid_define = """
        .......#.......
        ...............
        .......#.......
        .......#.......
        .......#.......
        .......#.......
        .......#......."""

        # Replace dots with 0 and hashtags with 1
        occupancy_grid_define = occupancy_grid_define.replace('.', '0').replace('#', '1')
        # Split the string into lines and remove any empty lines
        lines = [line.replace(' ', '') for line in occupancy_grid_define.split('\n') if line]
        # Convert each line into a list of integers
        grid = [list(map(int, line)) for line in lines]
        # Convert the list of lists into a 2D numpy array
        self.occupancy_grid = np.array(grid)
        return self.occupancy_grid
    
    def is_in_free_space(self, x_next, y_next):
        try:
            if self.occupancy_grid[y_next, x_next] == 0:
                return True
        except IndexError:
            return False
    
    def get_x_max(self):
        return self.occupancy_grid.shape[1]
    
    def get_y_max(self):
        return self.occupancy_grid.shape[0]
    

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
        #print(sampled_actions)
        
        #prune non-valid actions
        sampled_actions = [action for action in sampled_actions if env.is_in_free_space(self.x1 + action[0], self.y1 + action[1]) and env.is_in_free_space(self.x2 + action[0], self.y2 + action[1])]
        return sampled_actions
    
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
        if self.timestep >= horizon:
            return True
        return False
    
    # Terminal Reward
    def get_reward(self): 
        # TODO: possibly extending with projection onto centerline
        euclidean_distance = np.sqrt((self.x1 - self.x2)**2+(self.y1 - self.y2)**2)
        x_distance = self.x1 - self.x2
        progress_and_lead = self.x1 + x_distance
        distance_to_goal_state = np.exp(-abs(10-self.x1)-abs(5-self.y1))
        return int(progress_and_lead)
    
    def get_state(self):
        return [self.x1, self.y1, self.x2, self.y2, self.timestep]

class MCTSNode:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._sum_of_rewards_X = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        # create a list of two dictionaries for each agent to store the regrets of each action
        self._regrets = None
        self._regrets = self.build_regret_table()
        # print(self._regrets)
        
    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions
    
    ###################
    ### Regret Matching
    ###################

    def build_regret_table(self):
        # RM: creates a table with regrets for each action of each agent
        actions_agent0, actions_agent1 = self.state.get_legal_actions_seperate()

        _regrets_0 = pd.DataFrame({
        'Agent0_Actions': actions_agent0,
        'Agent0_Regrets': [0]*len(actions_agent0),
        })

        _regrets_1 = pd.DataFrame({
        'Agent1_Actions': actions_agent1,
        'Agent1_Regrets': [0]*len(actions_agent1),
        })

        return [_regrets_0, _regrets_1]
    
    def update_regrets(self, action, utility_01):
        #update the regrets fixing the actions of agent 0
        action_0 = action[:2]
        for index in self._regrets[0].index:
            action_to_update_0 = self._regrets[0].loc[index,'Agent0_Actions']

            child_temporal = self.get_child_regret_matching(action_0, action_to_update_0)
            utility_temporal = child_temporal.rollout()
            self._regrets[0].loc[index, 'Agent0_Regrets'] += (utility_temporal-utility_01)
        
        #update the regrets fixing the actions of agent 1
        action_1 = action[2:]
        for index in self._regrets[1].index:
            action_to_update_1 = self._regrets[1].loc[index,'Agent1_Actions']

            child_temporal = self.get_child_regret_matching(action_to_update_1, action_1)
            utility_temporal = -child_temporal.rollout() # negative because of symmetry of zero-sum games and definition of reward function
            self._regrets[1].loc[index, 'Agent1_Regrets'] += (utility_temporal-utility_01)
        #print(self._regrets)
        
    def get_child_regret_matching(self, action_0, action_1):
        action = action_0 + action_1
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        return child_node
    
    def best_action_regret_matching(self, gamma=0.5):
        R_plus_sum_0 = 0
        action_list_0 = [self._regrets[0]['Agent0_Actions'][index] for index in self._regrets[0].index]
        abs_A_0 = len(action_list_0)
        
        for index in self._regrets[0].index:
            R_plus_sum_0 += max(0, self._regrets[0]['Agent0_Regrets'][index])
        if R_plus_sum_0 <= 0:
            action_0_probs_from_regrets = [1/abs_A_0 for _ in range(abs_A_0)]
        else:
            action_0_probs_from_regrets = [max(0, self._regrets[0]['Agent0_Regrets'][index])/R_plus_sum_0 for index in self._regrets[0].index]
        
        action_0_probs_uniform = [1/abs_A_0 for _ in range(abs_A_0)]
        gamma_policy_0 = [gamma*a + (1-gamma)*b for a, b in zip(action_0_probs_uniform, action_0_probs_from_regrets)]

        best_action_0 = random.choices(action_list_0, weights=gamma_policy_0)[0]
        #print("Action List 0: {}".format(action_list_0))
        #print("Best action 0: {}".format(best_action_0))

        R_plus_sum_1 = 0
        action_list_1 = [self._regrets[1]['Agent1_Actions'][index] for index in self._regrets[1].index]
        abs_A_1 = len(action_list_1)
        for index in self._regrets[1].index:
            R_plus_sum_1 += max(0, self._regrets[1]['Agent1_Regrets'][index])
        if R_plus_sum_1 <= 0:
            action_1_probs_from_regrets = [1/abs_A_1 for _ in range(abs_A_1)]
        else:
            action_1_probs_from_regrets = [max(0, self._regrets[1]['Agent1_Regrets'][index])/R_plus_sum_1 for index in self._regrets[1].index]
        
        action_1_probs_uniform = [1/abs_A_1 for _ in range(abs_A_1)]
        gamma_policy = [gamma*a + (1-gamma)*b for a, b in zip(action_1_probs_uniform, action_1_probs_from_regrets)]
        best_action_1 = random.choices(action_list_1, weights=gamma_policy)[0]

        return best_action_0 + best_action_1
    
    def best_child_regret_matching(self, gamma=0.5):
        best_action = self.best_action_regret_matching(gamma)
        for child in self.children:
            if child.parent_action == best_action:
                return child

    
    def n(self):
        return self._number_of_visits
    
    def expand(self):
        action = self._untried_actions.pop() # TODO: which action to pop, maybe random?
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node
    
    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    def rollout(self):
        current_rollout_state = self.state

        while not current_rollout_state.is_terminal():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.get_reward()
    
    def backpropagate(self, utility):
        # backpropagate statistics of the node
        self._number_of_visits += 1
        self._sum_of_rewards_X += utility
        if self.parent:
            self.parent.backpropagate(utility)
    
    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = self.best_child_regret_matching(gamma=0.5)
        return current_node
    
    def next_node(self):
        simulation_count = 100

        for _ in range(simulation_count):
            v = self._tree_policy()
            reward = v.rollout()
            v.update_regrets(v.parent_action, reward)
            v.backpropagate(reward)
        print(v._regrets)
        return self.best_child_regret_matching(gamma=0.5)

def plot_trajectory(trajectory, map):
        # plotting the trajectory of two agents on a 2D-plane and connecting states with a line and labeling states with timestep
        # trajectory: list of states [x1, y1, x2, y2, timestep]
        fig, ax = plt.subplots()
        trajectory_x_0 = [trajectory[i][0] for i in range(len(trajectory))]
        trajectory_y_0 = [trajectory[i][1] for i in range(len(trajectory))]
        trajectory_x_1 = [trajectory[i][2] for i in range(len(trajectory))]
        trajectory_y_1 = [trajectory[i][3] for i in range(len(trajectory))]
        timesteps = [trajectory[i][4] for i in range(len(trajectory))]
        ax.plot(trajectory_x_0, trajectory_y_0, "bo-", label='Trajectory Agent 0')
        ax.plot(trajectory_x_1, trajectory_y_1, "ro-", label='Trajectory Agent 1')
        # annotate timesteps
        for i in range(len(trajectory)):
            ax.annotate(timesteps[i], (trajectory_x_0[i], trajectory_y_0[i]), textcoords="offset points", xytext=(0,10))
            ax.annotate(timesteps[i], (trajectory_x_1[i], trajectory_y_1[i]), textcoords="offset points", xytext=(0,10))
        ax.legend()
        ax.imshow(map)
        plt.show()

if __name__ == "__main__":

    env = Environment()
    map = env.occupancy_grid
    
    horizon = 5
    intial_state = State(0, 6, 0, 4, 0)
    root = MCTSNode(intial_state)
    current_node = root

    trajectory = [intial_state.get_state()]

    for h in range(horizon):
        current_node = current_node.next_node()
        trajectory.append(current_node.state.get_state())
        print(trajectory)
    
    plot_trajectory(trajectory, map)


    