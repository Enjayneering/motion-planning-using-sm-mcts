from subconf_mcts_vs_mcts import agent_conf_0, agent_conf_1, model_conf, mctsEE_0, mctsEE_1
from environments import track2 as env

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    confdict_0 = {"agent_conf_0": agent_conf_0.confdict,
                  "agent_conf_1": agent_conf_1.confdict,
              "model_conf": model_conf.confdict,
              "algo_conf_0": mctsEE_0.confdict,
              "algo_conf_1": mctsEE_1.confdict,
              "env_conf": env.confdict,}
    experiments.append({'name': 'racetrack_V2', 'dict': confdict_0})
    return experiments