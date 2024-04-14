from subconf import agent_conf_0, agent_conf_1, model_conf, mcts_0, mcts_1
from environments import street as env
import os

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    confdict_0 = {"agent_conf_0": agent_conf_0.confdict,
                  "agent_conf_1": agent_conf_1.confdict,
              "model_conf": model_conf.confdict,
              "algo_conf_0": mcts_0.confdict,
              "algo_conf_1": mcts_1.confdict,
              "env_conf": env.confdict,}
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    foldername = os.path.basename(curr_dir)
    experiments.append({'name': foldername, 'dict': confdict_0})
    return experiments