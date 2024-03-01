import numpy as np

# import modules
import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS')
from utilities.environment_utilities import get_closest_gridpoint
from mcts import *
from environment import *


#TODO: to be done

class Agent:
    def __init__(self, agent_ix, curr_timestep, max_timestep, config_agents, config_environment):
        # set perspective of the agent, so that the agent's index is always 0
        self.agent_indices = get_index_persepective(agent_ix, config_agents)
        self.agent_env = Environment(config_environment, self.agent_indices)
        self.kinematic_params = assume_kinematic_params(self.agent_indices, config_agents)
        self.config_mcts = config_agents[agent_ix]["config_mcts"]
        

        # initialize start and goal state
        self.curr_timestep = curr_timestep
        self.max_timestep = max_timestep
        # current state
        self.curr_joint_state = self.agent_env.joint_init_state

    def update_current_joint_state(self, sensed_joint_state_global):
        # transforming global state in local indexing and updateing the current state
        sensed_joint_state = sensed_joint_state_global[self.agent_indices]
        self.curr_joint_state = sensed_joint_state
 
    def compute_next_step(self, game_timehorizon):

        ####################
        # Choose algorithm #
        ####################
        mp_algorithm = MCTS(self.curr_joint_state, self.curr_timestep, self.max_timestep, self.config_mcts)
        
        next_joint_state, runtime, policies = mp_algorithm.compute_next_step(game_timehorizon)
        
        self.curr_joint_state = next_joint_state
        return next_joint_state, runtime, policies
    
def assume_kinematic_params(agent_indices, config_agents):
    """
    transform global dict data to local list storing discrete values for each agent
    """
    kinematic_params = []
    for agent_ix in agent_indices:
        kinematic_params.append([])
        kinematic_params[-1].append(config_agents[agent_ix]['assumptions'][agent_ix]['vel'])
        kinematic_params[-1].append(config_agents[agent_ix]['assumptions'][agent_ix]['ang_vel'])
    return kinematic_params


def get_index_persepective(agent_index, config_agents):
    """
    returns the new order of indices of all agents so that the agent can reshuffle data to its perspective [agent_ix, ...]
    """
    agents = np.arange(len(config_agents))
    shifted_agents = np.roll(agents, -agent_index)
    return shifted_agents
