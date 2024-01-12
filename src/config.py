from dataclasses import dataclass
import numpy as np

experimental_mode = False

feature_flags={
                'final_move': {'robust': True, 'best': False},
                'collision_handling': {'punishing': True, 'pruning': False},
                'selection_policy': {'ucb': True, 'random': False},
                'rollout_policy': {'random': True, 'best': False},
                'payoff_weights': {'fixed': True, 'adaptive': False},
            }

@dataclass
class Config:
    # Configure Environment
        env_name: str
        env_def: dict
        theta_0_init: float
        theta_1_init: float

        # Hyperparameter Simulation
        num_sim: int

        # Hyperparameter MCTS
        num_iter: int
        max_timehorizon: int
        delta_t: float
        c_param: float

        # Payoff weights
        penalty_distance_0: float
        penalty_distance_1: float
        reward_progress_0: float
        reward_progress_1: float
        penalty_agressor_0: float
        penalty_agressor_1: float
        penalty_timestep_0: float
        penalty_timestep_1: float
        reward_lead_0: float
        reward_lead_1: float

        # Action Sets
        velocity_0: list
        ang_velocity_0: list
        velocity_1: list
        ang_velocity_1: list

        # feature flags
        feature_flags: dict = None

# Flexible test configuration
test_config = Config(
            env_name = 'test',
            env_def = {0:"""
                        ################
                        #........#######
                        #..........#####
                        #0.1...........#
                        #..........#####
                        #........#######
                        ################""",
                        
                        },
            theta_0_init = 0,
            theta_1_init = 0,
            num_sim=1,
            num_iter=800,
            max_timehorizon=12, #[t]=s #ggf 6-8
            delta_t = 0.5, #[dt]=s
            c_param = np.sqrt(2),
            penalty_distance_0=-1,
            penalty_distance_1=-1,
            reward_progress_0=0,
            reward_progress_1=0,
            penalty_agressor_0=0,
            penalty_agressor_1=0,
            penalty_timestep_0=-1,
            penalty_timestep_1=-1,
            reward_lead_0=5,
            reward_lead_1=5,
            velocity_0=np.linspace(0, 2, 3).tolist(),
            ang_velocity_0=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            velocity_1=np.linspace(0, 1, 2).tolist(),
            ang_velocity_1=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            )

# Experiment001: Vary weights
exp001_config = Config(
            env_name = 'benchmark_static_small',
            env_def = {0:""" 
                            ################
                            #..............#
                            #..............#
                            #0.1...........#
                            #..............#
                            #..............#
                            ################""",
                            1:"""
                            ################
                            #..............#
                            #.1............#
                            #0.............#
                            #..............#
                            #..............#
                            ################""",
                            },
            theta_0_init = 0,
            theta_1_init = 0,
            num_sim=100,
            num_iter=1000,
            max_timehorizon=10,
            delta_t = 1,
            c_param = np.sqrt(2),
            penalty_distance_0=-2,
            penalty_distance_1=-5,
            reward_progress_0=1,
            reward_progress_1=1,
            penalty_agressor_0=-10,
            penalty_agressor_1=-10,
            penalty_timestep_0=-0.1,
            penalty_timestep_1=-0.1,
            reward_lead_0=5,
            reward_lead_1=5,
            velocity_0=np.linspace(0, 2, 3).tolist(),
            ang_velocity_0=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            velocity_1=np.linspace(0, 1, 2).tolist(),
            ang_velocity_1=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            feature_flags={
                            'final_move': {'robust': True, 
                                           'best': False},
                            'collision_handling': {'punishing': True, 
                                                   'pruning': False},
                            'selection_policy': {'ucb': True, 
                                                 'random': False},
                            'rollout_policy': {'uniform_random': True, 
                                               'best': False},
                            'payoff_weights': {'adaptive': False, 
                                               'fixed': True},
                        }
            )




def is_feature_active(feature_flag):
    return feature_flag
