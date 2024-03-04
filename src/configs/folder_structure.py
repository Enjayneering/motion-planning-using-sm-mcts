import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
src_dir = os.path.dirname(current_dir)
main_dir = os.path.dirname(src_dir)

results_dir = os.path.join(main_dir, "results")

