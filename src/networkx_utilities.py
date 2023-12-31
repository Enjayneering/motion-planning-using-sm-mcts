import networkx as nx
import pickle
import networkx as nx
import pickle

def save_tree_to_file(node_current_timestep, path_to_tree):
    tree = nx.DiGraph()
    tree.add_node(node_current_timestep)

    stack = [node_current_timestep]
    while stack:
        current_node = stack.pop()
        for child in current_node.children:
            tree.add_edge(current_node, child)
            stack.append(child)

    with open(path_to_tree, 'wb') as file:
        pickle.dump(tree, file)

def open_tree_from_file(latest_file):
    # Create a new DiGraph tree
    #tree = nx.DiGraph()

    # Unpickle and load the tree from the file
    with open(latest_file, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        tree = unpickler.load()
    #print("Tree loaded from file: {}".format(tree))
    return tree
    
    