from subconf import default_config, duct_config
from subconf.config_utilities import copy_new_dict
from environments import intersection as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    global_exp_dir = "01_intersection"

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, duct_config.config_dict, env.confdict)
    experiments.append({'name': 'duct', 
                        'timestep_sim': None,
                        'dict': dict_0,
                        'exp_params': {'num_iter': (100, 500, 10),
                                       'weight_final': (0.0, 0.1, 11),}})
    return experiments, global_exp_dir


