from .subconf_mcts_vs_mcts import agent_conf, model_conf, mcts_conf
from .environments import intersection as env_conf

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    confdict_0 = {"agent_conf": agent_conf.confdict,
              "model_conf": model_conf.confdict,
              "algo_conf_0": mcts_conf.confdict,
              "algo_conf_1": mcts_conf.confdict,
              "env_conf": env_conf.confdict,}
    
    # basic experiment
    experiments.append({'name': 'run_0', 'dict': confdict_0}) 
    return experiments