from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from dataset.datasets import DatasetKind
from dataset.hard_policy import HardPolicy
from dataset.main import (
    get_bbox_of,
    load_clean_labels,
    load_test_set,
    run_mode_classes,
    run_mode_kfoldcv,
    run_mode_trainval,
)

ALLOW_TEST_SET = True


class NuclearCataractDataset:
    class KFoldCVMode:
        k_folds: int
        fold_mapping: dict

        def __init__(self, k_folds: int):
            self.k_folds = k_folds

    class TrainValMode:
        train_prop: float
        val_prop: float
        split_mapping: dict

        def __init__(self, train_prop: float, val_prop: float):
            self.train_prop = train_prop
            self.val_prop = val_prop

    class TestMode:
        samples: list[dict[str, str]]

    def __init__(
        self,
        mode: KFoldCVMode | TrainValMode | TestMode,
        cache_size: int | None = None,
        return_paths: bool = False,
        return_bboxes: bool = False,
        hard_policy: HardPolicy = HardPolicy.PASSTHROUGH,
        dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
    ):
        self.mode = mode
        self.cache_size = cache_size
        self.return_paths = return_paths
        self.return_bboxes = return_bboxes
        self.dataset_kind = dataset_kind

        if isinstance(mode, NuclearCataractDataset.TrainValMode):
            mode.split_mapping = run_mode_trainval(
                mode.train_prop,
                mode.val_prop,
                should_flatten_packs=True,
                hard_policy=hard_policy,
                dataset_kind=self.dataset_kind,
            )
        elif isinstance(mode, NuclearCataractDataset.KFoldCVMode):
            mode.fold_mapping = run_mode_kfoldcv(
                mode.k_folds,
                should_flatten_packs=True,
                hard_policy=hard_policy,
                dataset_kind=self.dataset_kind,
            )
        elif isinstance(mode, NuclearCataractDataset.TestMode):
            mode.samples = load_test_set(  # type: ignore
                should_flatten_packs=True,
                hard_policy=hard_policy,
                dataset_kind=self.dataset_kind,
            )
        else:
            raise NotImplementedError

        self.label_to_idx = run_mode_classes(False, self.dataset_kind)
        self.n_classes = len(self.label_to_idx)
        self.label_names = {k for k in self.label_to_idx}

    def train_set(self):
        if isinstance(self.mode, NuclearCataractDataset.TrainValMode):
            return NuclearCataractSubset(
                self.mode.split_mapping["train"],
                self.label_to_idx,
                self.return_paths,
                self.return_bboxes,
                self.cache_size is not None,
                self.cache_size,
                self.dataset_kind,
            )
        else:
            raise NotImplementedError

    def val_set(self):
        if isinstance(self.mode, NuclearCataractDataset.TrainValMode):
            return NuclearCataractSubset(
                self.mode.split_mapping["val"],
                self.label_to_idx,
                self.return_paths,
                self.return_bboxes,
                self.cache_size is not None,
                self.cache_size,
                self.dataset_kind,
            )
        else:
            raise NotImplementedError

    def fold_train_set(self, fold_idx: int):
        if isinstance(self.mode, NuclearCataractDataset.KFoldCVMode):
            samples = [
                s
                for j in range(self.mode.k_folds)
                if j != fold_idx
                for s in self.mode.fold_mapping[str(j)]
            ]
            return NuclearCataractSubset(
                samples,
                self.label_to_idx,
                self.return_paths,
                self.return_bboxes,
                self.cache_size is not None,
                self.cache_size,
                self.dataset_kind,
            )
        else:
            raise NotImplementedError

    def fold_val_set(self, fold_idx: int):
        if isinstance(self.mode, NuclearCataractDataset.KFoldCVMode):
            return NuclearCataractSubset(
                self.mode.fold_mapping[str(fold_idx)],
                self.label_to_idx,
                self.return_paths,
                self.return_bboxes,
                self.cache_size is not None,
                self.cache_size,
                self.dataset_kind,
            )
        else:
            raise NotImplementedError

    def test_set(self):
        if not ALLOW_TEST_SET:
            raise RuntimeError(
                "You should probably not use test set, this may be changed after end of experiments"
            )

        if isinstance(self.mode, NuclearCataractDataset.TestMode):
            return NuclearCataractSubset(
                self.mode.samples,
                self.label_to_idx,
                self.return_paths,
                self.return_bboxes,
                self.cache_size is not None,
                self.cache_size,
                self.dataset_kind,
            )
        else:
            raise RuntimeError("test_set() can only be executed in TestMode")


class NuclearCataractSubset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        label_to_idx: dict[str, int],
        return_paths=False,
        return_bboxes=False,
        should_cache=True,
        cache_size: int | None = None,
        dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
    ):
        super().__init__()

        self.samples = samples
        self.label_to_idx = label_to_idx
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.should_cache = should_cache
        self.return_paths = return_paths
        self.return_bboxes = return_bboxes
        self.dataset_kind = dataset_kind
        self.cache = {}

        if return_bboxes:
            labels_df = load_clean_labels(dataset_kind)
            # Stores a list of boxes in XYXX format.
            # Each box is a list of four floats scaled from 0 to 1.
            # xmin,ymin is top left and
            # xmax,ymax is bottom right
            # Converted to a torch tensor for performance
            self.bboxes = [
                torch.tensor(
                    get_bbox_of(labels_df, Path(s["path"]).name),
                    dtype=torch.float32,
                ).reshape(-1, 4)
                for s in self.samples
            ]

        resize = v2.Resize(cache_size) if cache_size else None

        if should_cache:
            print("Loading images to cache...")

            for sample in tqdm(self.samples):
                img = self.__load_img(sample["path"])
                img = resize(img) if resize else img
                self.cache[sample["path"]] = img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.should_cache:
            result = [self.cache[sample["path"]], sample["label"]]
        else:
            result = [self.__load_img(sample["path"]), sample["label"]]

        result[1] = self.label_to_idx[result[1]]

        if self.return_paths:
            result.append(sample["path"])

        if self.return_bboxes:
            result.append(self.bboxes[idx])

        return result

    def __load_img(self, path: Path) -> torch.Tensor:
        # converts to RGB if the images have alpha channel
        return torchvision.io.decode_image(
            str(self.dataset_kind.root / path), mode=torchvision.io.ImageReadMode.RGB
        )

    def class_weights(self) -> torch.Tensor:
        n_classes = len(self.label_to_idx)
        classes_counts = torch.Tensor(self.__classes_counts())
        n = sum(classes_counts)
        return n / (n_classes * classes_counts)

    def __classes_counts(self) -> list[int]:
        classes_counts = [0 for _ in self.label_to_idx]

        for sample in self.samples:
            classes_counts[self.label_to_idx[sample["label"]]] += 1

        return classes_counts


if __name__ == "__main__":
    print("Running smoke test...")
    ncd = NuclearCataractDataset(
        NuclearCataractDataset.TrainValMode(0.8, 0.2),
        cache_size=224,
        return_paths=True
    )
    ss = ncd.train_set()
    print(ss[0])
    print(ss.class_weights())

    print("Running test set mode...")
    ALLOW_TEST_SET = True
    ncd = NuclearCataractDataset(NuclearCataractDataset.TestMode())
    ss = ncd.test_set()
    print(ss[0])

    print("Running gabinet test set mode...")
    ncd = NuclearCataractDataset(
        NuclearCataractDataset.TestMode(),
        return_paths=True,
        return_bboxes=True,
        dataset_kind=DatasetKind.GABINET,
    )
    ss = ncd.test_set()
    img, label, path, bboxes = ss[0]
    print(img.shape, img.dtype, img.min().item(), img.max().item(), img.float().mean().item())
    assert len(ss) > 0
    assert img.shape[0] == 3
    assert label in ncd.label_to_idx.values()
    assert bboxes.shape[1] == 4
    print(len(ss), path, bboxes)
