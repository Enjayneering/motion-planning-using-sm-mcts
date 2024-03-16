import os
import sys

# Get the current script's directory
src_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
main_dir = os.path.dirname(src_dir)

results_dir = os.path.join(main_dir, "results")
data_dir = os.path.join(main_dir, "data")

# experimental file structure
path_to_data = data_dir
