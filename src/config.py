from pathlib import Path
import json
import yaml
import numpy as np

"""
Provides generic Config class useful for passing around parameters
Also provides basic saving/loading functionality to and from json/yaml
"""
class Config(dict):
    """
    Wraps a dictionary to have its keys accesible like attributes
    I.e can do both config['steps'] and config.steps to get/set items
    Note - Recursively applies this class wrapper to each dict inside dict
    I.e:
    >>> config = Config({'filepath': '/path/to/file',
                         'settings': {'a': 1, 'b': 2}})
    >>> print(config.filepath)
    /path/to/file
    >>> config.settings.c = 3
    >>> print(config.settings.c, config.settings['c'])
    3 3
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = Config(v)

    def __getattr__(self, key):
        if key not in self:
            raise KeyError(f"key doesn't exist: {key}")
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __delattr__(self, name):
        if name in self:
            del self[name]
        raise AttributeError("No such attribute: " + name)

    @classmethod
    def from_path(cls, filepath):
        """ Loads config from given filepath """
        filepath = Path(filepath)
        if filepath.suffix == '.json':
            return cls(load_config_json(filepath))
        if filepath.suffix == '.yaml':
            return cls(load_config_yaml(filepath))
        raise NotImplementedError("Only .json or .yaml extensions supported")

    def save(self, savepath):
        """
        Saves config as either json or yaml depending on suffix
        of savepath provided.
        """
        savepath = Path(savepath)
        if savepath.suffix == '.json':
            save_config_json(self, savepath)
        elif savepath.suffix == '.yaml':
            save_config_yaml(self, savepath)
        raise NotImplementedError("Only .json or .yaml extensions supported")

    def as_dict(self):
        return config_to_dict(self)


def copy_new_config(default_config, new_dict):
    new_config = Config(default_config)
    for key, value in new_dict.items():
        if key in new_config:
            new_config[key] = value
        else:
            new_config.__dict__[key] = value
    return new_config

def config_to_dict(config: Config):
    """
    Recursively converts config objects to dict
    """
    config = dict(config)
    for k, v in config.items():
        if isinstance(v, Config):
            config[k] = config_to_dict(v)
    return config


def load_config_json(path):
    with open(path, 'r') as f:
        out = json.load(f)
    return Config(out)


def save_config_json(config, path, write_mode='w'):
    with open(path, write_mode) as f:
        json.dump(dict(config), f)


def load_config_yaml(path):
    with open(path, 'r') as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
    return Config(conf)


def save_config_yaml(config, path, write_mode='w'):
    config = config_to_dict(config)
    with open(path, write_mode) as f:
        yaml.dump(config, f)

def is_feature_active(feature_flag):
    return feature_flag


default_dict = {
    'env_name': str,
    'env_def': dict,
    'theta_0_init': 0,
    'theta_1_init': 0,
    'terminal_progress': float,
    'alpha_t': 1, # defines ratio of game timehorizon with respect to a heuristic estimator for timesteps needed to reach the end of the track
    'num_sim': 1,
    'num_iter': 1000,
    'delta_t': 1,
    'c_param': np.sqrt(2),
    'penalty_distance_0': float,
    'penalty_distance_1': float,
    'reward_progress_0': float,
    'reward_progress_1': float,
    'penalty_agressor_0': float,
    'penalty_agressor_1': float,
    'penalty_timestep_0': float,
    'penalty_timestep_1': float,
    'reward_lead_0': float,
    'reward_lead_1': float,
    'velocity_0': list,
    'ang_velocity_0': list,
    'velocity_1': list,
    'ang_velocity_1': list,
    'feature_flags': {'final_move': {'robust': True, 'best': False},
                'collision_handling': {'punishing': True, 'pruning': False},
                'selection_policy': {'ucb': True, 'random': False},
                'rollout_policy': {'random': True, 'best': False},
                'payoff_weights': {'fixed': True, 'adaptive': False},
                'expansion_policy': {'full_child': True, 'random': False},
                },
    }

test_dict = {
    'env_name': 'test',
    'env_def': {
        0: """
            ########################
            #1.....................#
            #####.....##############
            ###.....################
            #..0...#################
            ########################""",
    },
    'theta_0_init': 0,
    'theta_1_init': 0,
    'terminal_progress': 20,
    'alpha_t': 1,
    'num_sim': 1,
    'num_iter': 400,
    'delta_t': 1,
    'penalty_distance_0': -0.2,
    'penalty_distance_1': -0.2,
    'reward_progress_0': 0,
    'reward_progress_1': 0,
    'penalty_agressor_0': 0,
    'penalty_agressor_1': 0,
    'penalty_timestep_0': -0.2,
    'penalty_timestep_1': -0.2,
    'reward_lead_0': 1,
    'reward_lead_1': 1,
    'velocity_0': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'feature_flags': {'final_move': {'robust': True, 'best': False},
                'collision_handling': {'punishing': True, 'pruning': False},
                'selection_policy': {'ucb': True, 'random': False},
                'rollout_policy': {'random': True, 'best': False},
                'payoff_weights': {'fixed': True, 'adaptive': False},
                'expansion_policy': {'full_child': True, 'random': False},
                },
    
}

exp001_dict = {
    'env_name': 'closing-door',
    'env_def': {
        0: """
            ################
            #..............#
            #..............#
            #0.1...........#
            #..............#
            #..............#
            ################""",
        2: """
            ################
            #.......#......#
            #..............#
            #..............#
            #..............#
            #..............#
            ################""",
        4: """
            ################
            #.......#......#
            #.......#......#
            #..............#
            #..............#
            #..............#
            ################""",
        6: """
            ################
            #.......#......#
            #.......#......#
            #.......#......#
            #..............#
            #..............#
            ################""",

    },
    'theta_0_init': 0,
    'theta_1_init': 0,
    'terminal_progress': 12,
    'alpha_t': 1,
    'num_sim': 1,
    'num_iter': 100,
    'delta_t': 1,
    'penalty_distance_0': -2,
    'penalty_distance_1': -5,
    'reward_progress_0': 1,
    'reward_progress_1': 1,
    'penalty_agressor_0': -10,
    'penalty_agressor_1': -10,
    'penalty_timestep_0': -0.1,
    'penalty_timestep_1': -0.1,
    'reward_lead_0': 5,
    'reward_lead_1': 5,
    'velocity_0': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 1, 2).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'feature_flags': {
        'final_move': {'robust': True, 'best': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'ucb': True, 'random': False},
        'rollout_policy': {'uniform_random': True, 'best': False},
        'payoff_weights': {'adaptive': False, 'fixed': True},
        'expansion_policy': {'full_child': True, 'random': False},
    }
}

experimental_mode = True

default_config = Config(default_dict)
test_config = copy_new_config(default_config, test_dict)
exp001_config = copy_new_config(default_config, exp001_dict)

