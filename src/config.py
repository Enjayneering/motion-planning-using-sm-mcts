import numpy as np
from config_utilities import *

env_dict = {
    'racetrack-7x16': {
        'env_def': {
            0: """
                ################
                ................
                ................
                .0.1............
                ................
                ................
                ################""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
        'finish_line': 12,
    },
    'racetrack-three-gaps': {
        'env_def': {
            0: """
                ##############
                ..............
                .......#......
                .......#......
                .0.....#......
                ..............
                .1.....#......
                .......#......
                .......#......
                ..............
                ##############""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
        'finish_line': 12,
    },
    'racetrack-one-gap': {
        'env_def': {
            0: """
                ########
                .0..#...
                ........
                .1..#...
                ########""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
        'finish_line': 6,
    },
    'racetrack-4x13': {
        'env_def': {
            0: """
                #############
                ..0..........
                1............
                #############""",
            3: """
                #############
                ....+........
                ....+........
                #############""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
        'finish_line': 12,
    },
    'racetrack-closing-doors': {
        'env_def': {
            0: """
                ##############
                ####0#........
                ####..........
                ####.#........
                .1............
                ##############""",
            5: """
                ##############
                ####.#........
                ####.+........
                ####.#........
                .....+........
                ##############""",
        },
        'theta_0_init': np.pi/2,
        'theta_1_init': 0,
        'finish_line': 12,
    },
    'overtaking_no_obs': {
        'env_def': {
            0: """
                ################
                .0...............
                ...............
                .1...............
                ################""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
        'finish_line': 12,
    },
    'lane_merging': {
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
        'finish_line': 12,
    },
    'closing_door': {
        'env_def': {
            0: """
                ###############
                .0.....########
                ...............
                .1.....########
                ###############""",
            5: """
                ###############
                .......########
                .......+.......
                .......########
                ###############""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
        'finish_line': 12,
    },
}

default_dict = {
    'terminal_progress': float, # estimator of game length based on tracksize and agents maximum speed
    'rollout_length': int, # length of rollout trajectory
    'alpha_t': 1, # defines ratio of game timehorizon with respect to a heuristic estimator for timesteps needed to reach the end of the track
    'final_move_depth': 1,

    'num_sim': 1,
    'num_iter': 1000,
    'delta_t': 1,
    'c_param': np.sqrt(2),

    'collision_distance': 0.5,

    'collision_ignorance': 0.5,
    'discount_factor': 0.9,

    'penalty_distance': float,
    'reward_progress_0': float,
    'reward_progress_1': float,
    'penalty_timestep_0': float,
    'penalty_timestep_1': float,
    'reward_lead_0': float,
    'reward_lead_1': float,

    'velocity_0': list,
    'ang_velocity_0': list,
    'velocity_1': list,
    'ang_velocity_1': list,

    'standard_dev_vel_0': float,
    'standard_dev_ang_vel_0': float,
    'standard_dev_vel_1': float,
    'standard_dev_ang_vel_1': float,
    'feature_flags': {
        'final_move': {'robust-joint': False, 'robust-separate': True, 'depth-robust-joint': False, 'depth-robust-separate': False, 'max': False, 'ucb': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'ucb': True, 'max': False, 'regret-matching': False},
        'rollout_policy': {'random-uniform': True, 'random-informed': False},
        'payoff_weights': {'fixed': True, 'adaptive': False},
        'expansion_policy': {'every-child': True, 'random': False},
    }
    }
default_config = Config(default_dict)


exp_overtaking_punishcoll = {
    'c_param': np.sqrt(2),

    'final_move_depth': None,
    'num_final_move_childs': None,

    'rollout_length': 8,
    'terminal_progress': 8,

    'alpha_t': 1,
    'num_sim': 5,
    'num_iter': 1000,
    'delta_t': 1,

    'collision_ignorance': 0.4, #[0,1]
    'discount_factor': 0.8,

    'penalty_distance': -0,
    'reward_progress_0': 0.5,
    'reward_progress_1': 0.5,
    'penalty_timestep_0': -0.05,
    'penalty_timestep_1': -0.05,
    'reward_lead_0': 1,
    'reward_lead_1': 1,
    'velocity_0': np.linspace(0, 1, 2).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 1, 2).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),

    'standard_dev_vel_0': 1,
    'standard_dev_ang_vel_0': np.pi/4,
    'standard_dev_vel_1': 1,
    'standard_dev_ang_vel_1':  np.pi/4,

    'feature_flags': {
        'final_move': {'robust-joint': True, 'robust-separate': False, 'depth-robust-joint': False, 'depth-robust-separate': False, 'max': False, 'ucb': False},
        'collision_handling': {'punishing': False, 'pruning': True},
        'selection_policy': {'ucb': True, 'max': False, 'regret-matching': False},
        'rollout_policy': {'random-uniform': True, 'random-informed': False},
        'payoff_weights': {'fixed': True, 'adaptive': False},
        'expansion_policy': {'every-child': True, 'random': False},
    }
}

experimental_mode = True 



