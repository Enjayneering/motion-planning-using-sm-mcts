import numpy as np


config_dict = {
    'name': 'exp3',

    # MCTS Parameters
    'c_param': np.sqrt(2),
    'k_samples': 1,
    'num_iter': 1000,
    'gamma_exp3': 0.1,

    # Engineering Parameters
    'alpha_rollout': 1,
    'alpha_terminal': 1.0,
    'delta_t': 1,
    
    # Statistical Analysis
    'num_sim': 10,

    # Payoff Parameters
    'discount_factor': 0.9,

    'weight_interm': 0.5,
    'weight_final': 1,
    
    #interm payoffs
    'weight_distance': 0,
    'weight_collision': 1,
    'weight_progress': 0,
    'weight_lead': 0,

    # final payoffs
    'weight_timestep': 0,
    'weight_winning': 0, # better not, because too ambiguous
    'weight_final_lead': 1,

    # Behavioural Parameters
    'collision_ignorance': 0.5, #[0,1]
    
    'velocity_0': np.linspace(0, 1, 2).tolist(),
    'ang_velocity_0': [0], #np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 1, 2).tolist(),
    'ang_velocity_1': [0], #np.linspace(-np.pi/2, np.pi/2, 3).tolist(),

    'standard_dev_vel_0': 2,
    'standard_dev_ang_vel_0': np.pi/2,
    'standard_dev_vel_1': 1,
    'standard_dev_ang_vel_1':  np.pi/2,

    'feature_flags': {
        'run_mode': {'test': False, 'exp': True, 'live-plot': True},
        'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'uct-decoupled': False, 'regret-matching': False, 'exp3': True},
        'rollout_policy': {'random-uniform': True, 'random-informed': False},
        'expansion_policy': {'every-child': True, 'random-informed': False},
        'strategy': {'pure': True, 'mixed': False},
    }
}
