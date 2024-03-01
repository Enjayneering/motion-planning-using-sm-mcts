import networkx as nx
import pickle
import networkx as nx
import pickle

# import modules
import os 
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.insert(0, parent_dir)

from common_utilities import *

#TODO: change to plot each agents solutions in subfolder

def init_tree_file(test_id, agent_ix,  path_to_trees):
    new_path = path_to_tree.format(test_id, agent_ix)
    # Delete all existing tree files
    file_list = os.listdir(new_path)
    for file_name in file_list:
        file_path = os.path.join(new_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_tree_to_file(node_current_timestep, path_to_tree):
    tree = nx.DiGraph()
    tree.add_node(node_current_timestep)

    stack = [node_current_timestep]
    while stack:
        current_node = stack.pop()
        for child in current_node.children:
            tree.add_edge(current_node, child)
            stack.append(child)

    with open(path_to_tree, 'wb') as f:
        pickle.dump(tree, f)

def open_tree_from_file(latest_file):
    # Unpickle and load the tree from the file
    with open(latest_file, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        tree = unpickler.load()
    #print("Tree loaded from file: {}".format(tree))
    return tree
    
    