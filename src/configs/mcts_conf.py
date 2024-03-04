mcts_conf = {
    0: {
        'num_iter': 1000,
        'gamma_exp3': 0.2,

        # Payoff Parameters
        'discount_factor': 0.9,
        'weight_interm': 0.5,
        'weight_final': 1,
        
        'weight_distance': 0,
        'weight_collision': 1,
        'weight_progress': 0,
        'weight_lead': 0, # better not, too complex

        'weight_timestep': 0,
        'weight_winning': 0, # better not, because too ambiguous
        'weight_final_lead': 1,

        'feature_flags': {
                'selection_policy': {0: {'uct-decoupled': True, 'regret-matching': False, 'exp3': False, 'predefined_trajectory': False}, 
                                     1: {'uct-decoupled': True, 'regret-matching': False, 'exp3': False, 'predefined_trajectory': False}},
                'expansion_policy': {'every-child': True, 'random-informed': False},
            }
    },
    1: {
        'num_iter': 1000,
        'gamma_exp3': 0.2,

        # Payoff Parameters
        'discount_factor': 0.9,
        'weight_interm': 0.5,
        'weight_final': 1,
        
        'weight_distance': 0,
        'weight_collision': 1,
        'weight_progress': 0,
        'weight_lead': 0, # better not, too complex

        'weight_timestep': 0,
        'weight_winning': 0, # better not, because too ambiguous
        'weight_final_lead': 1, 

        'feature_flags': {
                'selection_policy': {0: {'uct-decoupled': True, 'regret-matching': False, 'exp3': False, 'predefined_trajectory': False}, 
                                     1: {'uct-decoupled': True, 'regret-matching': False, 'exp3': False, 'predefined_trajectory': False}},
                'expansion_policy': {'every-child': True, 'random-informed': False},
            }
    }
}