import networkx as nx
import pickle
import networkx as nx
import pickle
import os

from common import *
import networkx as nx
import pickle
import os
from common import *

def init_tree_file():
    # Delete all existing tree files
    file_list = os.listdir(path_to_trees)
    for file_name in file_list:
        file_path = os.path.join(path_to_trees, file_name)
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
    
    