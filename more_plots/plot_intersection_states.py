import os
import pickle

def open_tree_from_file(latest_file):
    # Unpickle and load the tree from the file
    with open(latest_file, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        print(unpickler)
        tree = unpickler.load()
    #print("Tree loaded from file: {}".format(tree))
    return tree


if __name__ == "__main__":
    # Path to the tree
    tree_0 = os.path.join(os.path.dirname(__file__), "tree_0.csv")

    # Load the tree
    intersection_tree = open_tree_from_file(tree_0)

    print(intersection_tree)
