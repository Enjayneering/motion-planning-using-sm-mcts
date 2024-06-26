# This interface provides a way to use the MCTS algorithm as a receding horizon controller for a competitive game.
# It can be called from a supersccript to initialize the game and run the MCTS algorithm.
import numpy as np
import pandas as pd
from pandas import json_normalize 

from .bimatrix_mcts.bimatrix_mcts import BiMatrixMCTS
from .bimatrix_mcts.utilities.plot_test_utilities import *
from .bimatrix_mcts.utilities.plot_utilities import *
from .bimatrix_mcts.utilities.common_utilities import *
from .bimatrix_mcts.utilities.csv_utilities import *
from .bimatrix_mcts.utilities.run_utilities import *
from .bimatrix_mcts.utilities.environment_utilities import *
from .bimatrix_mcts.utilities.config_utilities import *



def run_bimatrix_interface(ix_agent, curr_state, env_conf, agent_conf, model_conf, algo_conf, timestep_sim=1):
    # change environment perspective
    
    #env_conf_agent = env_conf.copy()
    algo_conf = algo_conf[0] # somehow stored in a tuple...



    """if ix_agent == 1:
        # change perspective
        temp = env_conf_agent['env_raceconfig']['0']
        env_conf_agent['env_raceconfig']['0'] = env_conf_agent['env_raceconfig']['1']
        env_conf_agent['env_raceconfig']['1'] = temp
        curr_state = curr_state[3:6]+curr_state[0:3]+[curr_state[-1]] # [x1, y1, theta1, x0, y0, theta0, timestep]"""
    
    # Set all mcts configurations for working as controller for agent
    game_dict =  {
        'name': env_conf['name'],
        'collision_distance': model_conf['collision_distance'],

        # MCTS Parameters
        'c_param': np.sqrt(2),
        'k_samples': 1, # not further investigated here
        'num_iter': algo_conf['num_iter'],
        'gamma_exp3': algo_conf['gamma_exp3'],

        # Engineering Parameters
        'alpha_rollout': 1,
        'alpha_terminal': algo_conf['alpha_terminal'],
        'delta_t': model_conf['delta_t'],
        
        # Statistical Analysis -> NO
        'num_sim': 1,

        # Payoff Parameters
        'discount_factor': algo_conf['discount_factor'],
        'weight_interm': algo_conf['weight_interm'],
        'weight_final': algo_conf['weight_final'],
        
        'weight_distance': algo_conf['weight_distance'],
        'weight_collision': algo_conf['weight_collision'],
        'weight_progress': algo_conf['weight_progress'],
        'weight_lead': algo_conf['weight_lead'],

        'weight_timestep': algo_conf['weight_timestep'],
        'weight_winning': algo_conf['weight_winning'], # better not, because too ambiguous
        'weight_final_lead': algo_conf['weight_final_lead'],

        # Behavioural Parameters
        'collision_ignorance': 0.5, #[0,1] # Don't change
        
        'velocity_0': agent_conf[0]['velocity'],
        'ang_velocity_0': agent_conf[0]['ang_velocity'],
        'velocity_1': agent_conf[1]['velocity'],
        'ang_velocity_1': agent_conf[1]['ang_velocity'],

        'standard_dev_vel_0': np.max(agent_conf[0]['velocity']),
        'standard_dev_ang_vel_0': np.max(agent_conf[0]['ang_velocity']),
        'standard_dev_vel_1': np.max(agent_conf[1]['velocity']),
        'standard_dev_ang_vel_1': np.max(agent_conf[1]['ang_velocity']),

        'feature_flags': {
            'run_mode': {'test': False, 'exp': True, 'live-plot': True},
            'final_move': {'robust-joint': False, 'robust-separate': True, 'max': False},
            'collision_handling': {'punishing': True, 'pruning': False},
            'selection_policy': algo_conf['feature_flags']['selection_policy'],
            'rollout_policy': {'random-uniform': False, 'random-informed': True},
            'expansion_policy': algo_conf['feature_flags']['expansion_policy'],
            'strategy': algo_conf['feature_flags']['strategy'],
        },
        'predef_traj': algo_conf['predef_traj'],

    }
    
    # only run MCTS one single time
    print("\nNew Self-Play for agent {} at state {} \n".format(ix_agent, curr_state))

    Game_for_agent = BiMatrixMCTS(Environment(Config(env_conf)), Config(game_dict), init_state=curr_state)
    # Run MCTS
    result_dict, policy_df, sampled_trajectories = Game_for_agent.run_game(timestep_sim) #timestep sim =1



    
    algo_data = {'result_dict': result_dict, 'policy_df': policy_df}

    # Transform back to original perspective
    """if ix_agent == 1:
        # change perspective
        next_joint_state = result_dict['trajectory_1'][-1][:-1]+result_dict['trajectory_0'][-1][:-1] # without timestep [x,y,theta]
    else:"""
    if ix_agent == 0:
        next_state = result_dict['trajectory_0'][-1][:-1]
    elif ix_agent == 1:
        next_state = result_dict['trajectory_1'][-1][:-1]

    #print("\nMCTS for agent {}  chose the state {}\n".format(ix_agent, next_state))

    return next_state, algo_data