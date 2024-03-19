from subconf import default_config, exp3_config
from subconf.config_utilities import copy_new_dict
from environments import intersection as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    global_exp_dir = "04_intersection"

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, exp3_config.config_dict, env.confdict)
    experiments.append({'name': 'exp3', 
                        'timestep_sim': None,
                        'dict': dict_0,
                        'exp_params': {'num_iter': (100, 500, 10),
                                       'gamma_exp3': (0.1, 0.1, 4),}})
    return experiments, global_exp_dir


