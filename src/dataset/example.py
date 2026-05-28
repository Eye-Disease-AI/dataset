from pprint import pprint

import dataset.main


def main():
    print("10-fold CV:")
    subset_mapping = dataset.main.run_mode_kfoldcv(10)
    pprint(subset_mapping.keys())
    pprint(len(subset_mapping["0"]))

    print("Full mapping:")
    full_mapping = dataset.main.run_mode_kfoldcv(0)
    pprint(full_mapping.keys())


if __name__ == "__main__":
    main()
