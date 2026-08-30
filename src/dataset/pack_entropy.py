from pathlib import Path
import numpy as np
from main import split_trainval_set, load_test_set, load_clean_labels, DATA_BASE_PATH, \
    PATIENTS_OURS_JSON_PATH, count_unlabeled_packs, load_unlabeled_packs, SubsetSplitter, \
    generate_split_mapping, count_labeled_packs, download_files
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats
import json
import zipfile


ORIGINAL_SPLIT_PROPS = [0.7, 0.2, 0.1]
TRAINVAL_PROPS = [0.8, 0.2]
OLD_DATASET_DIR_PATH = DATA_BASE_PATH / "Nuclear_Cataract_Old"
OLD_DATASET_ZIP_NAME = "Nuclear_Cataract_2025_10_25.zip"
OLD_LABELS_PATH = OLD_DATASET_DIR_PATH / "Nuclear Cataract" / "labels.json"


def plot_subsets_entropies_new(dest_path: Path):
    plot_subsets_entropies(
        "Entropie rozkładu etykiet paczek danych (po poprawkach)",
        load_new_subsets(),
        dest_path,
    )
    print(f"Plotted new entropies @ {dest_path}")


def plot_subsets_entropies_old(dest_path: Path):
    plot_subsets_entropies(
        "Entropie rozkładu etykiet paczek danych (przed poprawkami)",
        load_old_subsets(),
        dest_path,
    )
    print(f"Plotted old entropies @ {dest_path}")


def plot_subsets_entropies(title: str, subsets: list[list[list[dict]]], save_path: Path):
    plt.figure()
    plt.title(title)
    plt.axis("off")
    plt.subplot(3, 1, 1)
    plot_subset_entropies(subsets[0])
    plt.subplot(3, 1, 2)
    plot_subset_entropies(subsets[1])
    plt.subplot(3, 1, 3)
    plot_subset_entropies(subsets[2])
    plt.savefig(save_path)
    plt.close()


# {"train": [[{"path": "...", "label": "..."}], ...], "val": ...}
def load_new_subsets() -> list[list[list[dict]]]:
    _, sm = split_trainval_set([0.8, 0.2], ["train", "val"])
    return [
        sm["train"],
        sm["val"],
        load_test_set(),
    ]


def load_old_subsets() -> list[list[list[dict]]]:
    prepare_old_dataset()
    labels_df = load_clean_labels(OLD_LABELS_PATH)

    with open(PATIENTS_OURS_JSON_PATH) as patients_ours:
        patients_ours_map = json.load(patients_ours)

    packs = load_unlabeled_packs(patients_ours_map)
    counted_packs = count_unlabeled_packs(packs, labels_df)

    trainval_test_ss = SubsetSplitter(
        counted_packs, 
        [ORIGINAL_SPLIT_PROPS[0] + ORIGINAL_SPLIT_PROPS[1], ORIGINAL_SPLIT_PROPS[2]],
    )
    trainval_test_sm = generate_split_mapping(
        labels_df, trainval_test_ss.subsets, ["trainval", "test"],
    )

    trainval_packs = count_labeled_packs(trainval_test_sm["trainval"])
    trainval_ss = SubsetSplitter(trainval_packs, TRAINVAL_PROPS)
    trainval_sm = generate_split_mapping(
        labels_df, trainval_ss.subsets, ["train", "val"],
    )

    return [
        trainval_sm["train"],
        trainval_sm["val"],
        trainval_test_sm["test"],
    ]


def prepare_old_dataset():
    if OLD_DATASET_DIR_PATH.exists():
        return

    zip_cache_path = download_files([OLD_DATASET_ZIP_NAME])[0]
    with zipfile.ZipFile(zip_cache_path, "r") as zip_ref:
        zip_ref.extractall(OLD_DATASET_DIR_PATH)


def plot_subset_entropies(subset: list[list[dict]]):
    entropies = []

    for pack in subset:
        entropy = calculate_pack_entropy(pack)
        entropies.append(entropy)

    plt.xlabel("Numer paczki danych")
    plt.ylabel("Entropia")
    plt.bar(range(len(entropies)), entropies)
    

def calculate_pack_entropy(pack):
    labels_counts = pack_labels_counts(pack)
    prob_dist = calculate_prob_dist(labels_counts)
    return calculate_dist_entropy(prob_dist)


def pack_labels_counts(pack):
    labels = []

    for sample in pack:
        labels.append(sample["label"])

    return list(Counter(labels).values())
    

def calculate_prob_dist(labels_counts):
    x = np.array(labels_counts)
    n = x.sum()
    return x / n
    

def calculate_dist_entropy(prob_dist):
    return scipy.stats.entropy(prob_dist)


if __name__ == "__main__":
    plot_subsets_entropies_new("entropies_new.png")
    plot_subsets_entropies_old("entropies_old.png")
