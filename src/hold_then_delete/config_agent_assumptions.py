import numpy as np
from hold_then_delete.config_utilities import *

config_agent_0 = {
    # Behavioural Parameters
    'collision_ignorance': 0.5, #[0,1]
    
    'velocity_0': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),

    'standard_dev_vel_0': 2,
    'standard_dev_ang_vel_0': np.pi/2,
    'standard_dev_vel_1': 1,
    'standard_dev_ang_vel_1':  np.pi/2,
}

config_agent_1 = {
    # Behavioural Parameters
    'collision_ignorance': 0.5, #[0,1]
    
    'velocity_0': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_0': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
    'velocity_1': np.linspace(0, 2, 3).tolist(),
    'ang_velocity_1': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),

    'standard_dev_vel_0': 2,
    'standard_dev_ang_vel_0': np.pi/2,
    'standard_dev_vel_1': 1,
    'standard_dev_ang_vel_1':  np.pi/2,
}

