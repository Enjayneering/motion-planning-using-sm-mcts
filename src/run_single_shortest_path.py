import time
import multiprocessing
import json
import itertools

from solvers.shortest_path.shortest_path_planner import ShortestPathPlanner

#from sm_mcts.environments import *
from utilities.env_utilities import Environment
from environments import intersection, street



SPPlanner = ShortestPathPlanner(Environment(street.env_conf), street.agent_conf, street.model_conf, street.mcts_conf)