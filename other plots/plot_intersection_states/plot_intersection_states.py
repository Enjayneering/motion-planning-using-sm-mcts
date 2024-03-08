import pickle
import os
import sys
import sys
sys.path.insert(0, '/home/enjay/0_thesis/01_MCTS/')
from src import solvers


def unpickle_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# file
curr_dir = os.path.dirname(__file__)
file_path = os.path.join(curr_dir, 'intersection_tree.csv')

# Use the function
data = unpickle_data(file_path)