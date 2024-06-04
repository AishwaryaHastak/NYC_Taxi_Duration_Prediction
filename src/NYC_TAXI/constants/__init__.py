from pathlib import Path

DATA_FILE_PATH = Path(r'C:\Users\h4has\Documents\Projects\NYC-Taxi\data')
OUTPUT_FILE_PATH = Path(r'.\outputs')

top_n = 5 # Number of top models that need to be evaluated to decide which one to promote

num_trials = 15 # The number of parameter evaluations for the optimizer to explore