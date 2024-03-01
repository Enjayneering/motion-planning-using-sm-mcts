import numpy as np
from config_colormap import config_colormap


config_agents = {
    # The `config_agents` dictionary is a configuration for multiple agents, where each agent is represented by a key-value pair.
    # The key is the agent's ID (0, 1, etc.), and the value is another dictionary containing the agent's configuration.
    # 
    # Each agent's configuration dictionary has the following keys:
    # - 'assumptions':  This is a dictionary where the keys are again the IDs of all agents and the values are dictionaries 
    #                   containing the beliefs of the current agent about the other agents. Each belief dictionary has 
    #                   keys 'vel' (velocity), 'ang_vel' (angular velocity), 'std_dev_vel' (standard deviation of velocity), 
    #                   'std_dev_ang_vel' (standard deviation of angular velocity), and 'color' (color of the agent in the visualization).
    # - 'color': This is the color of the current agent in the visualization.
    # - 'config_mcts': This is a dictionary containing the configuration for the Monte Carlo Tree Search (MCTS) algorithm used by the agent. 
    #                   It includes keys like 'final_move', 'collision_handling', 'selection_policy', 'rollout_policy', 'expansion_policy', 
    #                   'strategy', 'alpha_rollout', 'alpha_terminal', 'k_samples', 'c_param', and 'num_iter', each of which configures a 
    #                   different aspect of the MCTS algorithm.
    
    0:
        {
        'assumptions':  # model beliefs of agent 0
            {
                0: {'vel': [0,1,2],
                    'ang_vel': [-np.pi/2, 0, np.pi/2],
                    #'std_dev_vel': 2, ---> instead taking max of the velocity set
                    #'std_dev_ang_vel': np.pi/2,
                    },
                1: {'vel': [0,1], 
                    'ang_vel': [-np.pi/2, 0, np.pi/2],
                    #'std_dev_vel': 1,
                    #'std_dev_ang_vel': np.pi/2,
                    },
            },
        'color': config_colormap['mcts_red'],
        'config_mcts': 
            {
                'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
                'collision_handling': {'punishing': True, 'pruning': False},
                'selection_policy': {'DUCT': True, 'RM': False, 'EXP3': False},
                'rollout_policy': {'random-uniform': False, 'random-informed': True},
                'expansion_policy': {'every-child': True, 'random-informed': False},
                'strategy': {'pure': True, 'mixed': False},
                'alpha_rollout': 1,
                'alpha_terminal': 1.5,
                'k_samples': 1,
                'c_param': np.sqrt(2),
                'num_iter': 800,
            },
        },

    1:
        {
        'assumptions':  # model beliefs of agent 1
            {
            0: {'vel': [0,1,2],
                'ang_vel': [-np.pi/2, 0, np.pi/2],
                #'std_dev_vel': 2,
                #'std_dev_ang_vel': np.pi/2,
                },
            1: {'vel': [0,1], 
                'ang_vel': [-np.pi/2, 0, np.pi/2],
                #'std_dev_vel': 1,
                #std_dev_ang_vel': np.pi/2,
                },
            },
        'color': config_colormap['mcts_blue'],
        'config_mcts': 
            {
                'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
                'collision_handling': {'punishing': True, 'pruning': False},
                'selection_policy': {'DUCT': True, 'RM': False, 'EXP3': False},
                'rollout_policy': {'random-uniform': False, 'random-informed': True},
                'expansion_policy': {'every-child': True, 'random-informed': False},
                'strategy': {'pure': True, 'mixed': False},
                'alpha_rollout': 1,
                'alpha_terminal': 1.5,
                'k_samples': 1,
                'c_param': np.sqrt(2),
                'num_iter': 800,
            },
        },
}