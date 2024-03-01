import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS')

from setuptools import setup

setup(name='mcts_planner',
        version='0.1',
        description='MCTS Planner',
        install_requires=['numpy', 
                          'matplotlib', 
                          'networkx']
    )