import numpy as np
from hold_then_delete.config_utilities import *

default_dict = {

    'num_iter': 1000,
    'delta_t': 1,
    'c_param': np.sqrt(2),
    'k_samples': 1,

    'collision_distance': 0.5,
    'goal_distance': 1,

    'collision_ignorance': 0.5,
    'discount_factor': 0,

    'weight_distance': float,
    'weight_progress': float,
    'weight_collision': float,
    'weight_lead': float,

    'weight_timestep': float,
    'weight_winning': float,
    'weight_final_lead': float,

    'weight_interm': 1,
    'weight_final': 1,

    'velocity_0': list,
    'ang_velocity_0': list,
    'velocity_1': list,
    'ang_velocity_1': list,

    'standard_dev_vel_0': float,
    'standard_dev_ang_vel_0': float,
    'standard_dev_vel_1': float,
    'standard_dev_ang_vel_1': float,
    'feature_flags': {
        'run_mode': {'test': True, 'exp': False, 'live-plot': True},
        'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'uct-decoupled': True, 'cfr': False},
        'rollout_policy': {'random-uniform': True, 'random-informed': False},
        'payoff_weights': {'fixed': True, 'adaptive': False},
        'expansion_policy': {'every-child': True, 'random-informed': False},
        'strategy': {'pure': True, 'mixed': False},
    }
    }



config_agent_0 = {
    # Behavioural Parameters
    'collision_ignorance': 0.5, #[0,1]
    
    'velocity_0': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),

    'standard_dev_vel_0': 2,
    'standard_dev_ang_vel_0': np.pi/2,
    'standard_dev_vel_1': 1,
    'standard_dev_ang_vel_1':  np.pi/2,
}

