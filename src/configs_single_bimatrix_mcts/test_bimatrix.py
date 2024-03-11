from .subconf_test_bimatrix import default_config, duct_config
from .config_utilities import copy_new_dict
from .environments import dynamic_street as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, duct_config.config_dict, env.confdict)
    experiments.append({'name': 'exp3', 'dict': dict_0})

    # change action space agent 1
    dict_1 = copy_new_dict(default_config.default_dict, duct_config.config_dict, env.confdict)
    return experiments


