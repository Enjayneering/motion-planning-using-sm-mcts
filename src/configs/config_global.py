from sub_configs.config_agents import config_agents
from sub_configs.config_colormap import config_colormap
from hold_then_delete.config_mcts import config_mcts
from configs.sub_configs.config_payoffs import config_payoffs
from sub_configs.config_research import config_research
from sub_configs.config_environment import config_environment

config_global = {
    'state_space': ['x', 'y', 'theta'],
    'action_space': ['speed', 'angular_speed'],

    'delta_t': 1,
    'collision_distance': 0.5,
    'goal_distance': 1,

    'config_environment': config_environment,
    
    'num_sim_timesteps': None,

    'config_agents': config_agents,
    'config_colorspace': config_colormap,
    'config_payoffs': config_payoffs,
    
    'config_research': config_research,
}
