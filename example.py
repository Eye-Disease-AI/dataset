from pprint import pprint

import main

subset_mapping = main.run_mode_kfoldcv(10)
pprint(subset_mapping.keys())
pprint(len(subset_mapping["0"]))
