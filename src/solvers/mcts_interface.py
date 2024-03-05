# This interface provides a way to use the MCTS algorithm as a receding horizon controller for a competitive game.
# It can be called from a supersccript to initialize the game and run the MCTS algorithm.
import time
import multiprocessing
import json
import itertools
import numpy as np

from .sm_mcts.competitive_game import CompetitiveGame
from .sm_mcts.utilities.plot_test_utilities import *
from .sm_mcts.utilities.plot_utilities import *
from .sm_mcts.utilities.common_utilities import *
from .sm_mcts.utilities.csv_utilities import *
from .sm_mcts.utilities.run_utilities import *
from .sm_mcts.utilities.environment_utilities import *



def run_mcts_interface(ix_agent, curr_state, env_conf, agent_conf, model_conf, mcts_conf):
    # change environment perspective
    env_conf_agent = env_conf.copy()
    mcts_conf = mcts_conf[ix_agent]

    if ix_agent == 1:
        temp = env_conf_agent['env_raceconfig']['0']
        env_conf_agent['env_raceconfig']['0'] = env_conf_agent['env_raceconfig']['1']
        env_conf_agent['env_raceconfig']['1'] = temp
    
    # Set all mcts configurations for working as controller for agent
    game_dict =  {
        'env_name': env_conf['name'],
        'collision_distance': model_conf['collision_distance'],

        # MCTS Parameters
        'c_param': np.sqrt(2),
        'k_samples': 1,
        'num_iter': mcts_conf['num_iter'],
        'gamma_exp3': mcts_conf['gamma_exp3'],

        # Engineering Parameters
        'alpha_rollout': 1,
        'alpha_terminal': 1.5,
        'delta_t': model_conf['delta_t'],
        
        # Statistical Analysis -> NO
        'num_sim': 1,

        # Payoff Parameters
        'discount_factor': mcts_conf['discount_factor'],
        'weight_interm': mcts_conf['weight_interm'],
        'weight_final': mcts_conf['weight_final'],
        
        'weight_distance': mcts_conf['weight_distance'],
        'weight_collision': mcts_conf['weight_collision'],
        'weight_progress': mcts_conf['weight_progress'],
        'weight_lead': mcts_conf['weight_lead'],

        'weight_timestep': mcts_conf['weight_timestep'],
        'weight_winning': mcts_conf['weight_winning'], # better not, because too ambiguous
        'weight_final_lead': mcts_conf['weight_final_lead'],

        # Behavioural Parameters
        'collision_ignorance': 0.5, #[0,1] # Don't change
        
        'velocity_0': agent_conf[ix_agent]['velocity'],
        'ang_velocity_0': agent_conf[ix_agent]['ang_velocity'],
        'velocity_1': agent_conf[1-ix_agent]['velocity'],
        'ang_velocity_1': agent_conf[1-ix_agent]['ang_velocity'],

        'standard_dev_vel_0': np.max(agent_conf[ix_agent]['velocity']),
        'standard_dev_ang_vel_0': np.max(agent_conf[ix_agent]['ang_velocity']),
        'standard_dev_vel_1': np.max(agent_conf[1-ix_agent]['velocity']),
        'standard_dev_ang_vel_1': np.max(agent_conf[1-ix_agent]['ang_velocity']),

        'feature_flags': {
            'run_mode': {'test': False, 'exp': True, 'live-plot': False},
            'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
            'collision_handling': {'punishing': True, 'pruning': False},
            'selection_policy': mcts_conf['feature_flags']['selection_policy'],
            'rollout_policy': {'random-uniform': False, 'random-informed': True},
            'expansion_policy': mcts_conf['feature_flags']['expansion_policy'],
            'strategy': {'pure': True, 'mixed': False},
        }
    }

    timestep_sim = 1 # only run MCTS one single time
    Game_for_agent = CompetitiveGame(Environment(Config(env_conf_agent)), Config(game_dict), init_state=curr_state)

    # Run MCTS
    result_dict, policy_dict = Game_for_agent.run_game(timestep_sim)
    algo_data = {'result_dict': result_dict, 'policy_dict': policy_dict}

    next_joint_state = result_dict['trajectory_0'][-1][:-1]+result_dict['trajectory_1'][-1][:-1] # without timestep [x,y,theta]

    return next_joint_state, algo_data