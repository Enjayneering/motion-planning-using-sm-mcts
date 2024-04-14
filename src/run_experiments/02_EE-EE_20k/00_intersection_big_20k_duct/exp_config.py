from subconf_mcts_vs_mcts import agent_conf_0, agent_conf_1, model_conf, mctsEE_0, mctsEE_1
from environments import intersection

# SETTING EXPERIMENT UP
def build_experiments():
    experiments = []

    confdict_0 = {"agent_conf_0": agent_conf_0.confdict,
                  "agent_conf_1": agent_conf_1.confdict,
              "model_conf": model_conf.confdict,
              "algo_conf_0": mctsEE_0.confdict,
              "algo_conf_1": mctsEE_1.confdict,
              "env_conf": intersection.confdict,}
    experiments.append({'name': 'intersection_big_20k_duct', 'dict': confdict_0})
    return experiments