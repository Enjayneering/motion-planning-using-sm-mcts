import numpy as np

config_mcts = {
    0: {
            'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
            'collision_handling': {'punishing': True, 'pruning': False},
            'selection_policy': {'uct-decoupled': True, 'regret-matching': False, 'exp3': False},
            'rollout_policy': {'random-uniform': False, 'random-informed': True},
            'expansion_policy': {'every-child': True, 'random-informed': False},
            'strategy': {'pure': True, 'mixed': False},
            'alpha_rollout': 1,
            'alpha_terminal': 1.5,
            'k_samples': 1,
            'c_param': np.sqrt(2),
            'num_iter': 800,
        },
    1: {
            'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
            'collision_handling': {'punishing': True, 'pruning': False},
            'selection_policy': {'uct-decoupled': True, 'regret-matching': False, 'exp3': False},
            'rollout_policy': {'random-uniform': False, 'random-informed': True},
            'expansion_policy': {'every-child': True, 'random-informed': False},
            'strategy': {'pure': True, 'mixed': False},
            'alpha_rollout': 1,
            'alpha_terminal': 1.5,
            'k_samples': 1,
            'c_param': np.sqrt(2),
            'num_iter': 800,
        },
}