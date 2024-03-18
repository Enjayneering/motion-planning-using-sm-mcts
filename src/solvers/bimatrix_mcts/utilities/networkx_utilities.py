import networkx as nx
import pickle
import os

from .common_utilities import *

def init_tree_file():
    # Delete all existing tree files
    if not os.path.exists(path_to_trees):
        os.makedirs(os.path.dirname(path_to_trees), exist_ok=True)
    file_list = os.listdir(path_to_trees)
    for file_name in file_list:
        file_path = os.path.join(path_to_trees, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_tree_to_file(root_node, path_to_tree):
    tree = nx.DiGraph()
    tree.add_node(root_node)

    stack = [root_node]
    while stack:
        current_node = stack.pop()
        for child in current_node.children:
            tree.add_edge(current_node, child)
            stack.append(child)
    
    if not os.path.exists(path_to_trees):
        os.makedirs(os.path.dirname(path_to_trees), exist_ok=True)

    with open(path_to_tree, 'wb') as f:
        pickle.dump(tree, f)

def open_tree_from_file(latest_file):
    # Unpickle and load the tree from the file
    with open(latest_file, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        tree = unpickler.load()
    #print("Tree loaded from file: {}".format(tree))
    return tree
    
    