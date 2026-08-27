from pathlib import Path
import numpy as np
from main import split_trainval_set, load_test_set
import matplotlib.pyplot as plt
from collections import Counter
import scipy.stats


def plot_subsets_entropies():
    plt.figure()
    plt.title("Entropie rozkładu etykiet paczek danych")
    plt.axis("off")
    plt.subplot(3, 1, 1)
    plot_train_entropies()
    plt.subplot(3, 1, 2)
    plot_val_entropies()
    plt.subplot(3, 1, 3)
    plot_test_entropies()
    plt.savefig("entropies.png")
    plt.close()


def plot_train_entropies():
    subset = load_train_subset()
    plot_subset_entropies(subset, Path.cwd() / "train_entropies.png")


def plot_val_entropies():
    subset = load_val_subset()
    plot_subset_entropies(subset, Path.cwd() / "val_entropies.png")


def plot_test_entropies():
    subset = load_test_subset()
    plot_subset_entropies(subset, Path.cwd() / "test_entropies.png")


# {"train": [[{"path": "...", "label": "..."}], ...], "val": ...}
def load_train_subset() -> list[list[dict]]:
    _, sm = split_trainval_set([0.8, 0.2], ["train", "val"])
    return sm["train"]


def load_val_subset() -> list[list[dict]]:
    _, sm = split_trainval_set([0.8, 0.2], ["train", "val"])
    return sm["val"]
    

def load_test_subset() -> list[list[dict]]:
    return load_test_set()


def plot_subset_entropies(subset: list[list[dict]], dest_path: Path):
    entropies = []

    for pack in subset:
        entropy = calculate_pack_entropy(pack)
        entropies.append(entropy)

    print(entropies)
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
    plot_subsets_entropies()
