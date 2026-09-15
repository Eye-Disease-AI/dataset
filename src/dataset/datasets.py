from enum import StrEnum
from pathlib import Path
from typing import Literal

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_BASE_PATH = SCRIPT_DIR / "data"
GENERATED_DIR_PATH = DATA_BASE_PATH / "generated"

DatasetKindType = Literal["nuclear-cataract-ours", "gabinet"]


class DatasetKind(StrEnum):
    NUCLEAR_CATARACT: DatasetKindType = "nuclear-cataract-ours"
    GABINET: DatasetKindType = "gabinet"

    @property
    def root(self) -> Path:
        """Directory with datasets"""
        if self == DatasetKind.GABINET:
            return DATA_BASE_PATH / self.value / "data"
        elif self == DatasetKind.NUCLEAR_CATARACT:
            return DATA_BASE_PATH / self.value
        else:
            print(f"Unsupported dataset type {self}!")
            assert False

    @property
    def labels_path(self) -> Path:
        if self == DatasetKind.GABINET:
            return DATA_BASE_PATH / self.value / "labels.json"
        elif self == DatasetKind.NUCLEAR_CATARACT:
            return self.root / "labels.json"
        else:
            print(f"Unsupported dataset type {self}!")
            assert False

    @property
    def split_json_path(self) -> Path:
        """Trainval/test split, shipped with the dataset or generated on first use"""
        if self == DatasetKind.GABINET:
            return DATA_BASE_PATH / self.value / "split.json"
        elif self == DatasetKind.NUCLEAR_CATARACT:
            return GENERATED_DIR_PATH / "split.json"
        else:
            print(f"Unsupported dataset type {self}!")
            assert False
