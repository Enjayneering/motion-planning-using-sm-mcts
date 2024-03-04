import numpy as np
from .utilities.config_utilities import *

env_dict = {
    'racetrack_7x16_dis': {
        'env_def': {
            0: """
                ################
                ..............F.
                ..............F.
                x......x......Fx
                ..............F.
                ..............F.
                ################""",
        },
        'env_raceconfig': {
            '0': """
                ################
                ................
                ................
                .Sx..........xG.
                ................
                ................
                ################""",
            '1': """
                ################
                ................
                ................
                ...Sx........xG.
                ................
                ................
                ################""",
        },
    },
    'intersection_3x3': {
        'env_def': {
            0: """
                #.#
                ...
                #.#""",
        },
        'env_raceconfig': {
            '0':"""
                #S#
                .x.
                #G#""",
            '1':"""
                #.#
                SxG
                #.#""",
        },

    },
    'intersection_5x5': {
        'env_def': {
            0: """
                ##.##
                ##.##
                .....
                ##.##
                ##.##""",
        },
        'env_raceconfig': {
            '0':"""
                ##S##
                ##x##
                .....
                ##x##
                ##G##""",
            '1':"""
                ##.##
                ##.##
                Sx.xG
                ##.##
                ##.##""",
        },

    },
    'racetrack-three-gaps': {
        'env_def': {
            0: """
                ##############
                ............F.
                .......#....F.
                .......#....F.
                .0.....#....F.
                ............F.
                .1.....#....F.
                .......#....F.
                .......#....F.
                ............F.
                ##############""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
    },
    'racetrack-one-gap': {
        'env_def': {
            0: """
                ########
                .1..#...
                ........
                .0..#...
                ########""",
        },
        'theta_0_init': 0,
        'theta_1_init': 0,
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
    },
}

default_dict = {
    'env_name': str,
    'alpha_rollout': int, # percentage of rollout duration that is sampled randomly
    'alpha_terminal': 1, # defines ratio of game timehorizon with respect to a heuristic estimator for timesteps needed to reach the end of the track

    'num_sim': 1,
    'num_iter': 1000,
    'delta_t': 1,
    'c_param': np.sqrt(2),
    'k_samples': 1,

    'collision_distance': 0.5,
    'goal_distance': 1,

    'collision_ignorance': 0.5,
    'discount_factor': 0,

    'weight_distance': float,
    'weight_progress': float,
    'weight_collision': float,
    'weight_lead': float,

    'weight_timestep': float,
    'weight_winning': float,
    'weight_final_lead': float,

    'weight_interm': 1,
    'weight_final': 1,

    'velocity_0': list,
    'ang_velocity_0': list,
    'velocity_1': list,
    'ang_velocity_1': list,

    'standard_dev_vel_0': float,
    'standard_dev_ang_vel_0': float,
    'standard_dev_vel_1': float,
    'standard_dev_ang_vel_1': float,
    'feature_flags': {
        'run_mode': {'test': True, 'exp': False, 'live-plot': True},
        'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'uct-decoupled': True, 'cfr': False},
        'rollout_policy': {'random-uniform': True, 'random-informed': False},
        'payoff_weights': {'fixed': True, 'adaptive': False},
        'expansion_policy': {'every-child': True, 'random-informed': False},
        'strategy': {'pure': True, 'mixed': False},
    }
    }
default_config = Config(default_dict)


ff_exp_01 = {
        'run_mode': {'test': True, 'exp': False, 'live-plot': True},
        'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'uct-decoupled': True, 'regret-matching': False},
        'rollout_policy': {'random-uniform': False, 'random-informed': True},
        'expansion_policy': {'every-child': True, 'random-informed': False},
        'strategy': {'pure': True, 'mixed': False},
    }

ff_test_01 = {
        'run_mode': {'test': True, 'exp': False, 'live-plot': True},
        'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
        'collision_handling': {'punishing': True, 'pruning': False},
        'selection_policy': {'uct-decoupled': True, 'regret-matching': False},
        'rollout_policy': {'random-uniform': False, 'random-informed': True},
        'expansion_policy': {'every-child': True, 'random-informed': False},
        'strategy': {'pure': True, 'mixed': False},
    }


overtaking_dict = {
    'env_name': 'racetrack_7x16_dis',

    # MCTS Parameters
    'c_param': np.sqrt(2),
    'k_samples': 1,
    'num_iter': 800,

    # Engineering Parameters
    'alpha_rollout': 1,
    'alpha_terminal': 1.0,
    'delta_t': 1,
    
    # Statistical Analysis
    'num_sim': 10,

    # Payoff Parameters
    'discount_factor': 0.9,

    'weight_interm': 0.5,
    'weight_final': 1,
    
    #interm payoffs
    'weight_distance': 0,
    'weight_collision': 1,
    'weight_progress': 0,
    'weight_lead': 0,

    # final payoffs
    'weight_timestep': 0,
    'weight_winning': 0, # better not, because too ambiguous
    'weight_final_lead': 1,

    # Behavioural Parameters
    'collision_ignorance': 0.0, #[0,1]
    
    'velocity_0': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),

    'standard_dev_vel_0': 2,
    'standard_dev_ang_vel_0': np.pi/2,
    'standard_dev_vel_1': 1,
    'standard_dev_ang_vel_1':  np.pi/2,

    'feature_flags': ff_exp_01
}
