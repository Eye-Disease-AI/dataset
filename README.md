# dataset

Downloads, splits and loads slit lamp images for binary nuclear cataract
classification (`Zaćma` / `Brak Zaćmy`), with optional bounding boxes.

Two datasets are supported (`DatasetKind`):

| Kind | Value | Splits available |
|---|---|---|
| `NUCLEAR_CATARACT` | `nuclear-cataract-ours` | test + train/val + k-fold |
| `GABINET` | `gabinet` | test only (its `trainvalSet` is empty) |

## Concepts

- **Pack**: all images of one patient's eye (nuclear) or one visit (gabinet).
  Splitting is done per pack, so images of one eye never land in two subsets.
- **Hard pack**: a pack with mixed labels. `HardPolicy` controls them:
  `PASSTHROUGH` keeps all, `NO_HARD` drops hard packs, `ONLY_HARD` keeps only
  hard packs, `DOMINATE` relabels every image to the pack's majority label.
- **Split**: `split.json` holds `trainvalSet` and `testSet` as lists of packs.
  Train/val and k-fold subsets are deterministicly derived from `trainvalSet`
  at load time. `SubsetSplitter` assigns packs greedily to keep subset sizes,
  class balance and label entropy are close to the targets.


## Getting the dataset

The data is not in the repository. Get two archives from the project
maintainers and put them, unrenamed, in `.cache/` at the package root:

```
.cache/
  Nuclear_Cataract_2026_05_06.zip
  Gabinet_2026_09_15.zip
```

The gabinet archive is encrypted. Set its password, then load once:

```bash
export GABINET_ZIP_PASSWORD=...
uv sync
uv run python -m dataset.loader
```

The first load unpacks both archives into `data/`. Leave `data/` as it is
after cloning: the nuclear archive is only unpacked into a `data/` that holds
nothing but `.gitkeep`.

Resulting layout:

```
data/
  nuclear-cataract-ours/
    images/
    labels.json
  generated/
    split.json
  gabinet/
    data/            images, paths as referenced in labels.json
    labels.json
    split.json
```

An archive missing from `.cache/` is requested from a fileserver that no
longer exists, and loading fails.

## Preparing the dataset

```bash
# Regenerate split.json with train/val/test proportions (train and val are merged into trainvalSet)
uv run main -initial 0.7 0.2 0.1

# Write derived splits to data/generated/ and print split stats
uv run main -trainval 0.8 0.2
uv run main -kfoldcv 5
uv run main -classes 1
```

Regenerating the nuclear split changes the test set. Do not do it unless the
experiments are meant to move to a new split. It also needs the original
dataset in `data/nuclear-cataract-original/` and the unprocessed export in
`data/nuclear-cataract-ours/`, otherwise it tries to download them.

`-deps` downloads from the fileserver and will not work. `-zipgen 1` packs
`data/generated/` and `data/nuclear-cataract-ours/` into
`.out/Nuclear_Cataract_Generated.zip`, in the same layout as the archive above,
for handing the data to someone else. It requires `data/nuclear-cataract-original/` to exist.

Visual check (nuclear train set, 3x3 grid):

```bash
uv run visualize --bboxes --page 0
```

## Usage in code

```python
from dataset.datasets import DatasetKind
from dataset.hard_policy import HardPolicy
from dataset.loader import NuclearCataractDataset


ds = NuclearCataractDataset(
    NuclearCataractDataset.TrainValMode(0.8, 0.2),
    cache_size=224,
    dataset_kind=DatasetKind.NUCLEAR_CATARACT,
)
train, val = ds.train_set(), ds.val_set() # TrainValMode


K_FOLDS=5
fold_ds = NuclearCataractDataset(
    NuclearCataractDataset.KFoldCVMode(K_FOLDS),
    cache_size=224,
    dataset_kind=DatasetKind.NUCLEAR_CATARACT,
)
for i in range(K_FOLDS):
  fold_train, fold_val = ds.fold_train_set(i), ds.fold_val_set(i)


test_ds = NuclearCataractDataset(
    NuclearCataractDataset.TestMode(),
    cache_size=224,
    dataset_kind=DatasetKind.NUCLEAR_CATARACT,
)
test = ds.test_set()


img, label, *extra = train[0]
weights = train.class_weights() # inverse class frequency, for class weights in loss
```

Each subset is a `torch.utils.data.Dataset`.
An item is a list:

| Index | Content |
|---|---|
| 0 | `uint8` tensor `[3, H, W]`, RGB |
| 1 | `int` class index |
| next, if `return_paths` | path relative to the dataset root |
| last, if `return_bboxes` | `float32` tensor `[N, 4]`, xyxy normalized to 0-1, `N` may be 0 |

Bounding box count varies per image, so the default `DataLoader` collate
cannot stack them. Use a collate function that keeps boxes as a list.
Geometric augmentations must be applied to the boxes too.

Lower level functions in `dataset.main` (`run_mode_trainval`, `run_mode_kfoldcv`,
`load_test_set`, `load_clean_labels`, `get_bbox_of`) return the raw
`{"path", "label"}` mappings and label DataFrames.
