from .subconf_test_bimatrix import default_config, testconf
from .config_utilities import copy_new_dict
from .environments import street as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, testconf.config_dict, env.confdict)
    experiments.append({'name': 'test_bimatrix', 'dict': dict_0})

    # change action space agent 1
    dict_1 = copy_new_dict(default_config.default_dict, testconf.config_dict, env.confdict)
    return experiments


