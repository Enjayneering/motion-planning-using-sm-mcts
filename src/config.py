from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    # Configure Environment
        env_name_trigger: list

        # Hyperparameter Simulation
        num_sim: int

        # Hyperparameter MCTS
        num_iter: int
        max_timehorizon: int
        c_param: float

        # Payoff weights
        penalty_collision_0: float
        penalty_collision_1: float
        reward_progress_0: float
        reward_progress_1: float
        penalty_timestep_0: float
        penalty_timestep_1: float
        reward_lead_0: float
        reward_lead_1: float

        # Action Sets
        velocity_0: list
        ang_velocity_0: list
        velocity_1: list
        ang_velocity_1: list

exp_config = Config(
            env_name_trigger = [(0,'benchmark_dynamic_small2'), (1, 'benchmark_dynamic_small2_1'), (5, 'benchmark_dynamic_small2_2')],
            num_sim=100,
            num_iter=1000,
            max_timehorizon=10,
            c_param = np.sqrt(2),
            penalty_collision_0=-1,
            penalty_collision_1=-1,
            reward_progress_0=1,
            reward_progress_1=1,
            penalty_timestep_0=-0.05,
            penalty_timestep_1=-0.05,
            reward_lead_0=5,
            reward_lead_1=5,
            velocity_0=np.linspace(0, 2, 3).tolist(),
            ang_velocity_0=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            velocity_1=np.linspace(0, 1, 2).tolist(),
            ang_velocity_1=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            )

test_config = Config(
            env_name_trigger = [(0,'benchmark_dynamic_small2'), (1, 'benchmark_dynamic_small2_1'), (5, 'benchmark_dynamic_small2_2')],
            num_sim=1,
            num_iter=1000,
            max_timehorizon=12,
            c_param = np.sqrt(2),
            penalty_collision_0=-1,
            penalty_collision_1=-1,
            reward_progress_0=1,
            reward_progress_1=1,
            penalty_timestep_0=-1,
            penalty_timestep_1=-1,
            reward_lead_0=10,
            reward_lead_1=10,
            velocity_0=np.linspace(0, 2, 3).tolist(),
            ang_velocity_0=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            velocity_1=np.linspace(0, 1, 2).tolist(),
            ang_velocity_1=np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
            )

feature_flags = {
    'mode': {'experimental': False, 'test': True},
    'collision_pruning': False,
    'collision_punishing': True,
    'selection_seperate': True,
    'final_move_selection': {'robust': True, 'best': False},
}

def is_feature_active(feature_flag):
    return feature_flag
