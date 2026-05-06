from pprint import pprint

import main

print("10-fold CV:")
subset_mapping = main.run_mode_kfoldcv(10)
pprint(subset_mapping.keys())
pprint(len(subset_mapping["0"]))

print("Full mapping:")
full_mapping = main.run_mode_kfoldcv(0)
pprint(full_mapping.keys())
