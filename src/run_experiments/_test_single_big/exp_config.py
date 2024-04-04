from subconf import default_config, duct_config
from subconf.config_utilities import copy_new_dict
from environments import curve as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    global_exp_dir = "00_convergence_overtaking"

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, duct_config.config_dict, env.confdict)
    experiments.append({'name': 'big-race', 
                        'dict': dict_0,
                        })
    return experiments, global_exp_dir

