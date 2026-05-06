from pathlib import Path

import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from main import OURS_DATASET_PATH, split_trainval_set


class NuclearCataract(torch.utils.data.Dataset):
    """Class representing the dataset. Converts the dataset files into values usable to the model."""

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


class SubsetTransformer(torch.utils.data.Dataset):
    """Wrapper for subset that allows applying different transforms on each dataset subset (train, val, test)"""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        img = data[0]

        if self.transform is not None:
            img = self.transform(img)

        return tuple([img] + list(data[1:]))


_, split_mapping = split_trainval_set(
    [0.8, 0.2], ["train", "val"], should_flatten_packs=True
)
train_dataset = NuclearCataract(
    split_mapping["train"], cache_size=224, return_paths=True
)
print(train_dataset[0])
