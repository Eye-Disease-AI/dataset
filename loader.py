from pathlib import Path

import torch
import torchvision
from main import OURS_DATASET_PATH, load_test_set, run_mode_trainval
from torchvision.transforms import v2
from tqdm import tqdm

ALLOW_TEST_SET = False


class NuclearCataractDataset:
    class KFoldCVMode:
        k_folds: int

    class TrainValMode:
        train_prop: float
        val_prop: float
        split_mapping: dict

        def __init__(self, train_prop: float, val_prop: float):
            self.train_prop = train_prop
            self.val_prop = val_prop

    class TestMode:
        packs: list[dict]

    def __init__(
        self,
        mode: KFoldCVMode | TrainValMode | TestMode,
        cache_size: int | None = None,
        return_paths: bool = False,
    ):
        self.mode = mode
        self.cache_size = cache_size
        self.return_paths = return_paths

        if isinstance(mode, NuclearCataractDataset.TrainValMode):
            mode.split_mapping = run_mode_trainval(
                mode.train_prop, mode.val_prop, should_flatten_packs=True
            )
        elif isinstance(mode, NuclearCataractDataset.TestMode):
            mode.packs = load_test_set(should_flatten_packs=True)
        else:
            raise NotImplementedError

    def train_set(self):
        if isinstance(self.mode, NuclearCataractDataset.TrainValMode):
            return NuclearCataractSubset(
                self.mode.split_mapping["train"],
                self.return_paths,
                self.cache_size is not None,
                self.cache_size,
            )
        else:
            raise NotImplementedError

    def val_set(self):
        if isinstance(self.mode, NuclearCataractDataset.TrainValMode):
            return NuclearCataractSubset(
                self.mode.split_mapping["val"],
                self.return_paths,
                self.cache_size is not None,
                self.cache_size,
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
                self.mode.packs,
                self.return_paths,
                self.cache_size is not None,
                self.cache_size,
            )
        else:
            raise RuntimeError("test_set() can only be executed in TestMode")


class NuclearCataractSubset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples,
        return_paths=False,
        should_cache=True,
        cache_size: int | None = None,
    ):
        super().__init__()

        self.samples = samples
        self.should_cache = should_cache
        self.return_paths = return_paths
        self.cache = {}

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

        if self.return_paths:
            result.append(sample["path"])

        return result

    def __load_img(self, path: Path) -> torch.Tensor:
        return torchvision.io.decode_image(str(OURS_DATASET_PATH / path))


if __name__ == "__main__":
    print("Running smoke test...")
    ncd = NuclearCataractDataset(
        NuclearCataractDataset.TrainValMode(0.8, 0.2), cache_size=224, return_paths=True
    )
    ss = ncd.train_set()
    print(ss[0])

    print("Running test set mode...")
    ALLOW_TEST_SET = True
    ncd = NuclearCataractDataset(NuclearCataractDataset.TestMode())
    ss = ncd.test_set()
    print(ss[0])
