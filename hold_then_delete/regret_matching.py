 
    """def get_legal_actions_seperate(self):
        # returns a list of possible actions for each agent [[x1, y1], ...], [[x2, y2], ...]
        global env
        values = [-1, 0, 1]
        action_tuples = itertools.product(values, repeat=2)
        sampled_actions_0 = [list(action) for action in action_tuples]
        sampled_actions_1 = copy.deepcopy(sampled_actions_0)
    
        #prune non-valid actions
        sampled_actions_0 = [action for action in sampled_actions_0 if env.is_in_free_space(self.x1 + action[0], self.y1 + action[1])]
        sampled_actions_1 = [action for action in sampled_actions_1 if env.is_in_free_space(self.x2 + action[0], self.y2 + action[1])]

        return sampled_actions_0, sampled_actions_1"""


###################
### Regret Matching
###################

    """def build_regret_table(self):
        # RM: creates a table with regrets for each action of each agent
        actions_agent0, actions_agent1 = self.state.get_legal_actions_seperate()

        _regrets = pd.DataFrame({
        'Agent0_Actions': actions_agent0,
        'Agent0_Regrets': [0]*len(actions_agent0),
        })

        return _regrets
    
    def update_regrets(self, action, utility_01):
        #update the regrets fixing the actions of agent 0
        action_0 = action[:2]
        for index in self._regrets.index:
            action_to_update_0 = self._regrets.loc[index,'Agent0_Actions']

            child_temporal = self.get_child_regret_matching(action_0, action_to_update_0)
            utility_temporal = child_temporal.rollout()
            self._regrets.loc[index, 'Agent0_Regrets'] += (utility_temporal-utility_01)
        
    def get_child_regret_matching(self, action_0, action_1):
        action = action_0 + action_1
        next_state = self.state.move(action)
        child_node = MCTSNode(next_state, parent=self, parent_action=action)
        return child_node
    
    def best_action_regret_matching(self, gamma=0.5):
        R_plus_sum_0 = 0
        action_list_0 = [self._regrets['Agent0_Actions'][index] for index in self._regrets.index]
        abs_A_0 = len(action_list_0)
        
        for index in self._regrets.index:
            R_plus_sum_0 += max(0, self._regrets['Agent0_Regrets'][index])
        if R_plus_sum_0 <= 0:
            action_0_probs_from_regrets = [1/abs_A_0 for _ in range(abs_A_0)]
        else:
            action_0_probs_from_regrets = [max(0, self._regrets['Agent0_Regrets'][index])/R_plus_sum_0 for index in self._regrets.index]
        
        action_0_probs_uniform = [1/abs_A_0 for _ in range(abs_A_0)]
        gamma_policy_0 = [gamma*a + (1-gamma)*b for a, b in zip(action_0_probs_uniform, action_0_probs_from_regrets)]

        best_action_0 = random.choices(action_list_0, weights=gamma_policy_0)[0]
        #print("Action List 0: {}".format(action_list_0))
        #print("Best action 0: {}".format(best_action_0))

        rand_action_1 = random.choice(self.state.get_legal_actions_seperate()[1])

        return best_action_0 + rand_action_1
    
    def best_child_regret_matching(self, gamma=0.5):
        best_action = self.best_action_regret_matching(gamma)
        for child in self.children:
            if child.parent_action == best_action:
                return child"""