from subconf import default_config, duct_config
from subconf.config_utilities import copy_new_dict
from environments import street as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    global_exp_dir = "00_convergence_overtaking"

    # basic experiment
    dict_0 = copy_new_dict(default_config.default_dict, duct_config.config_dict, env.confdict)
    experiments.append({'name': 'test-mcts', 
                        'timestep_sim': None,
                        'dict': dict_0,
                        'exp_params': {}})

    return experiments, global_exp_dir

