confdict = {
    0: {
        'num_iter': 3100,
        'alpha_terminal': 1,
        'gamma_exp3': 0.2,

        # Payoff Parameters
        'discount_factor': 0.9,

        'weight_interm': 1,
        'weight_final': 1,
        
        'weight_distance': 0,
        'weight_collision': 1,
        'weight_progress': 0,
        'weight_lead': 0, # better not, too complex

        'weight_timestep': 0,
        'weight_winning': 0, # better not, because too ambiguous
        'weight_final_lead': 1,

        'feature_flags': {
                'selection_policy': {'uct-decoupled': True, 'regret-matching': False, 'exp3': False, 'predefined_trajectory': False}, 
                'expansion_policy': {'every-child': False, 'random-informed': True},
                'strategy': {'pure': True, 'mixed': False},
            }
    },
    1: {# assume strong
        'num_iter': 3100,
        'alpha_terminal': 1,
        'gamma_exp3': 0.2,

        # Payoff Parameters
        'discount_factor': 0.9,
        
        'weight_interm': 0.1,
        'weight_final': 0.9,
        
        'weight_distance': 0,
        'weight_collision': 1,
        'weight_progress': 0,
        'weight_lead': 0, # better not, too complex

        'weight_timestep': 0,
        'weight_winning': 0, # better not, because too ambiguous
        'weight_final_lead': 1, 

        'feature_flags': {
                'selection_policy': {'uct-decoupled': True, 'regret-matching': False, 'exp3': False, 'predefined_trajectory': False}, 
                'expansion_policy': {'every-child': False, 'random-informed': True},
                'strategy': {'pure': True, 'mixed': False},
            }
    }
}