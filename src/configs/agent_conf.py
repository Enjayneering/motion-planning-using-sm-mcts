import numpy as np

agent_conf = {
    0: {'velocity': np.linspace(0, 2, 3).tolist(),
        'ang_velocity': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
        },
    1: {'velocity': np.linspace(0, 1, 2).tolist(),
        'ang_velocity': np.linspace(-np.pi/2, np.pi/2, 3).tolist(),
        },
    
}