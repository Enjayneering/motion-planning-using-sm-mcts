from subconf import default_config, duct_config
from subconf.config_utilities import copy_new_dict
from environments import street as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    global_exp_dir = "00_convergence_following"

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, duct_config.config_dict, env.confdict)
    experiments.append({'name': 'duct-sampling_informed-expand_every', 
                        'timestep_sim': 1,
                        'dict': dict_0,
                        'exp_params': {'num_iter': (100, 500, 10),
                          'weight_interm': (0.0, 0.5, 3),
                          'alpha_terminal': (0.1, 0.2, 6),}})
    update_dict_1 = {'feature_flags': {
                    'run_mode': {'test': False, 'exp': True, 'live-plot': True},
                    'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
                    'collision_handling': {'punishing': True, 'pruning': False},
                    'selection_policy': {'uct-decoupled': True, 'regret-matching': False, 'exp3': False},
                    'rollout_policy': {'random-uniform': True, 'random-informed': False},
                    'expansion_policy': {'every-child': True, 'random-informed': False},
                    'strategy': {'pure': True, 'mixed': False},
                    }}
    dict_1 = copy_new_dict(dict_0, update_dict_1)
    experiments.append({'name': 'duct-sampling_uniform-expand_every', 
                        'timestep_sim': 1,
                        'dict': dict_1,
                        'exp_params': {'num_iter': (100, 500, 10),
                          'weight_interm':  (0.0, 0.5, 3),
                          'alpha_terminal': (0.1, 0.2, 6),}})
    return experiments, global_exp_dir


