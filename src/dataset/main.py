import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict
from getpass import getpass
from math import log
from pathlib import Path

import numpy as np
import pandas as pd
import pyzipper
import requests
import tqdm
from PIL import Image
from requests.auth import HTTPBasicAuth

from dataset.datasets import DATA_BASE_PATH, GENERATED_DIR_PATH, DatasetKind
from dataset.hard_policy import HardPolicy

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
ORIGINAL_DATASET_PATH = DATA_BASE_PATH / "nuclear-cataract-original"
OURS_DATASET_PATH = DatasetKind.NUCLEAR_CATARACT.root
OUR_IMAGES_DIR = OURS_DATASET_PATH / "images"
PATIENTS_ORIGINAL_JSON_PATH = GENERATED_DIR_PATH / "patients_original.json"
OUR_TO_ORIGINAL_JSON_PATH = GENERATED_DIR_PATH / "our_to_original.json"
ORIGINAL_TO_OUR_JSON_PATH = GENERATED_DIR_PATH / "original_to_our.json"
PATIENTS_OURS_JSON_PATH = GENERATED_DIR_PATH / "patients_ours.json"
KFOLDCV_JSON_PATH = GENERATED_DIR_PATH / "kfoldcv.json"
TRAINVAL_JSON_PATH = GENERATED_DIR_PATH / "trainval.json"
CLASSES_JSON_PATH = GENERATED_DIR_PATH / "classes.json"
CACHE_PATH = SCRIPT_DIR / ".cache"
FILESERVER_URL = "https://fileserver.krzyzanowski.dev"
ORIGINAL_DATASET_ZIP_NAME = "Nuclear_Cataract_Original.zip"
OURS_DATASET_ZIP_NAME = "Nuclear_Cataract_2025_12_21.zip"
PREPARED_DATASET_ZIP_NAME = "Nuclear_Cataract_2026_05_06.zip"
GABINET_ZIP_NAME = "Gabinet_2026_09_15.zip"
GABINET_ZIP_PASSWORD_ENV = "GABINET_ZIP_PASSWORD"
GABINET_DIAGNOSIS_MAPPING = {"Zdrowe": "Brak Zaćmy", "Zaćma": "Zaćma"}
GABINET_READABLE_QUALITY = "Czytelne"
GABINET_URL_PREFIX = "?d=gabinet/"
ZIPGEN_TMP_DIR = DATA_BASE_PATH / "Nuclear_Cataract"
OUTPUT_DIR = SCRIPT_DIR / ".out"
ZIPGEN_OUTPUT_ZIP_PATH_WITHOUT_EXT = OUTPUT_DIR / "Nuclear_Cataract_Generated"


class SubsetSplitter:
    EPSILON = 1e-9

    def __init__(
        self,
        all_count_packs: list,
        desired_props: list[float],
    ):
        self.desired_props = desired_props

        self.full_set = all_count_packs

        self.cataracts_count = 0
        self.non_cataracts_count = 0

        for pack in self.full_set:
            self.cataracts_count += pack["cataracts_count"]
            self.non_cataracts_count += pack["non_cataracts_count"]

        self.full_set_classes_counts = [self.cataracts_count, self.non_cataracts_count]

        self.subsets = [[] for _ in desired_props]

        # ASSUMPTION: 2 classes in the dataset
        self.subsets_classes_counts = [[0, 0] for _ in self.subsets]

        # Sum of probability distributions of datapacks in given subset
        self.subset_entropy_desired_props = [
            1.0 / len(self.subsets) for _ in self.subsets
        ]
        self.subset_entropy_sums = [0.0 for _ in self.subsets]

        for pack in self.full_set:
            self.__add_pack(pack)

    def get_subset_sizes_counts(self) -> list:
        result = []

        for subset in self.subsets:
            subset_size = 0
            for pack in subset:
                subset_size += len(pack["pack"])
            result.append(subset_size)

        return result

    def get_full_size(self) -> int:
        return sum(self.get_subset_sizes_counts())

    def get_subset_entropy_props(self) -> list:
        return self.__probabilize(self.subset_entropy_sums)

    def get_subset_sizes_props(self) -> list:
        subset_sizes = self.get_subset_sizes_counts()
        full_size = float(sum(subset_sizes))
        return [s / full_size for s in subset_sizes]

    def get_subset_sizes_desired_props(self) -> list:
        return self.desired_props

    def get_subset_sizes_desired_counts(self) -> list:
        full_size = self.get_full_size()
        result = [int(prop * full_size) for prop in self.desired_props]
        rest = full_size - sum(result)

        # Split out leftovers resulting from float inaccuracy
        for i in range(len(result)):
            if rest == 0:
                break
            result[i] += 1
            rest -= 1

        return result

    def get_subset_class_props(self) -> list:
        return [
            float(counts[0]) / (sum(counts) + SubsetSplitter.EPSILON)
            for counts in self.subsets_classes_counts
        ]

    def get_subset_class_desired_props(self) -> list:
        return [
            float(self.full_set_classes_counts[0]) / sum(self.full_set_classes_counts),
        ] * len(self.subsets)

    def __add_pack(self, count_pack: dict):
        subset_idx = self.__where_best(count_pack)
        pack_entropy = entropy(
            [
                count_pack["cataracts_count"],
                count_pack["non_cataracts_count"],
            ]
        )

        self.subsets[subset_idx].append(count_pack)
        self.subsets_classes_counts[subset_idx][0] += count_pack["cataracts_count"]
        self.subsets_classes_counts[subset_idx][1] += count_pack["non_cataracts_count"]
        self.subset_entropy_sums[subset_idx] += pack_entropy

    # ASSUMPTION: Subset balance equal to whole dataset balance is desired
    def __where_best(self, count_pack: dict) -> int:
        improvs = []

        for i, counts in enumerate(self.subsets_classes_counts):
            improv = self.__calc_improv(
                i, counts, self.full_set_classes_counts, count_pack
            )
            improvs.append(improv)

        return int(np.argmax(improvs))

    def __calc_improv(
        self,
        subset_idx: int,
        subset_classes_counts: list,
        full_set_classes_counts: list,
        count_pack: dict,
    ):
        cur_class_props_ce = SubsetSplitter.__ce(
            subset_classes_counts, full_set_classes_counts
        )
        new_class_props_ce = SubsetSplitter.__ce(
            [
                subset_classes_counts[0] + count_pack["cataracts_count"],
                subset_classes_counts[1] + count_pack["non_cataracts_count"],
            ],
            full_set_classes_counts,
        )

        class_props_improv = cur_class_props_ce - new_class_props_ce

        cur_counts = self.get_subset_sizes_counts()
        subset_size = sum(subset_classes_counts)

        new_counts = self.get_subset_sizes_counts()
        new_counts[subset_idx] += subset_size

        cur_subset_props_ce = SubsetSplitter.__ce(
            cur_counts,
            self.get_subset_sizes_desired_counts(),
        )
        new_subset_props_ce = SubsetSplitter.__ce(
            new_counts,
            self.get_subset_sizes_desired_counts(),
        )

        subset_sizes_improv = cur_subset_props_ce - new_subset_props_ce

        cur_entropy_props = self.get_subset_entropy_props()
        new_entropy_sums = self.subset_entropy_sums[:]
        new_entropy_sums[subset_idx] += count_pack["labels_entropy"]
        new_entropy_props = self.__probabilize(new_entropy_sums)

        cur_subset_entropy_props_ce = SubsetSplitter.__ce_props(
            cur_entropy_props,
            self.subset_entropy_desired_props,
        )
        new_subset_entropy_props_ce = SubsetSplitter.__ce_props(
            new_entropy_props,
            self.subset_entropy_desired_props,
        )

        subset_entropy_props_improv = (
            cur_subset_entropy_props_ce - new_subset_entropy_props_ce
        )

        return (
            class_props_improv + subset_sizes_improv + subset_entropy_props_improv
        ) / 2

    def print_stats(self):
        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        print("Subset sizes:", self.get_subset_sizes_counts())

        print("Class balance (actual):", self.get_subset_class_props())
        print("Class balance (desired):", self.get_subset_class_desired_props())

        print("Props (actual):", self.get_subset_sizes_props())
        print("Props (desired):", self.get_subset_sizes_desired_props())

        print("Hard examples / entropy (actual):", self.get_subset_entropy_props())
        print("Hard examples / entropy (desired):", self.subset_entropy_desired_props)

    @staticmethod
    def __ce_props(Q: list[float], P: list[float]) -> float:
        result = 0.0

        for q, p in zip(Q, P):
            result += -p * log(q + SubsetSplitter.EPSILON)

        return result

    @staticmethod
    def __ce(q_counts: list, p_counts: list) -> float:
        Q = [float(q) / (sum(q_counts) + SubsetSplitter.EPSILON) for q in q_counts]
        P = [float(p) / (sum(p_counts) + SubsetSplitter.EPSILON) for p in p_counts]

        return SubsetSplitter.__ce_props(Q, P)

    @staticmethod
    def __probabilize(p_counts: list) -> list[float]:
        return [float(p) / (sum(p_counts) + SubsetSplitter.EPSILON) for p in p_counts]


def original_dataset_filter(full_path: Path):
    name = full_path.name
    not_xslx = not name.endswith(".xlsx")
    not_dotfile = not name.startswith(".")
    return not_xslx and not_dotfile


def patient_dir_filter(full_path: Path):
    name = full_path.name
    not_datafile = name != "DATAFILE"
    not_dotfile = not name.startswith(".")
    return not_datafile and not_dotfile


def jpg_filter(full_path: Path):
    name = full_path.name
    return name.lower().endswith(".jpg")


def list_all_jpgs(dir_path: Path):
    jpgs_list = []

    for root, _, files in os.walk(dir_path):
        for file in files:
            jpgs_list.append(Path(root) / Path(file))

    return list(filter(jpg_filter, jpgs_list))


def generate_patients_original_mapping() -> dict[str, dict]:
    original_patients_dirs = list(ORIGINAL_DATASET_PATH.iterdir())
    original_patients_dirs = list(
        filter(original_dataset_filter, original_patients_dirs)
    )

    patients_map = {}

    # IZQ - left side
    # DER - right side

    for patient_dir_name in original_patients_dirs:
        patient_dir_path = ORIGINAL_DATASET_PATH / patient_dir_name

        eyes_dirs = list(patient_dir_path.iterdir())
        eyes_dirs_lower = list(map(lambda p: p.name.lower(), eyes_dirs))

        left_eyes_paths = []
        right_eyes_paths = []

        try:
            izq_dir_name = eyes_dirs[eyes_dirs_lower.index("izq")]
            izq_dir_path = patient_dir_path / izq_dir_name
            left_eyes_paths.extend(list_all_jpgs(izq_dir_path))
        except ValueError:
            pass

        try:
            der_dir_name = eyes_dirs[eyes_dirs_lower.index("der")]
            der_dir_path = patient_dir_path / der_dir_name
            right_eyes_paths.extend(list_all_jpgs(der_dir_path))
        except ValueError:
            pass

        patient_dir_name_rel_str = str(
            patient_dir_name.relative_to(ORIGINAL_DATASET_PATH)
        )
        patients_map[patient_dir_name_rel_str] = {
            "left": list(
                map(lambda p: str(p.relative_to(patient_dir_path)), left_eyes_paths)
            ),
            "right": list(
                map(lambda p: str(p.relative_to(patient_dir_path)), right_eyes_paths)
            ),
        }

    return patients_map


def save_patients_original_mapping(patients_original_map: dict[str, dict]):
    patients_images_json = json.dumps(patients_original_map, indent=4)
    with open(PATIENTS_ORIGINAL_JSON_PATH, "w+") as f:
        f.write(patients_images_json)


def get_original_images_paths(patients_original_map: dict[str, dict]) -> list[Path]:
    original_images_paths = []

    for patient_id in patients_original_map.keys():
        for image_path in (
            patients_original_map[patient_id]["left"]
            + patients_original_map[patient_id]["right"]
        ):
            original_images_paths.append(
                ORIGINAL_DATASET_PATH / patient_id / image_path
            )

    return original_images_paths


def load_our_images_to_memory() -> dict[Path, bytes]:
    our_images_paths = list(OUR_IMAGES_DIR.iterdir())
    our_images_bytes = {}

    for our_image_path in tqdm.tqdm(our_images_paths, desc="loading our images"):
        with open(our_image_path, "rb") as f:
            our_image_bytes = f.read()
        our_images_bytes[our_image_path] = our_image_bytes

    return our_images_bytes


def stringify_path_dict(d: dict[Path, Path]) -> dict[str, str]:
    return {str(k): str(v) for k, v in d.items()}


def generate_our_to_original_mappings(
    our_images_bytes: dict[Path, bytes], patients_original_map: dict[str, dict]
) -> tuple[dict[Path, Path], dict[Path, Path]]:
    original_images_paths = get_original_images_paths(patients_original_map)
    original_to_our = {}
    our_to_original = {}

    our_images_paths = list(OUR_IMAGES_DIR.iterdir())

    for original_image_path in tqdm.tqdm(
        original_images_paths, desc="comparing images"
    ):
        with open(original_image_path, "rb") as f:
            original_image_bytes = f.read()

        for our_image_path in our_images_paths:
            our_image_bytes = our_images_bytes[our_image_path]
            are_identical = original_image_bytes == our_image_bytes

            if are_identical:
                original_to_our[
                    original_image_path.relative_to(ORIGINAL_DATASET_PATH)
                ] = our_image_path.relative_to(OURS_DATASET_PATH)
                our_to_original[our_image_path.relative_to(OURS_DATASET_PATH)] = (
                    original_image_path.relative_to(ORIGINAL_DATASET_PATH)
                )
                our_images_paths.remove(our_image_path)
                break

    return original_to_our, our_to_original


def save_our_to_original_mappings(
    original_to_our: dict[Path, Path], our_to_original: dict[Path, Path]
):
    with open(ORIGINAL_TO_OUR_JSON_PATH, "w+") as f:
        f.write(json.dumps(stringify_path_dict(original_to_our), indent=4))

    with open(OUR_TO_ORIGINAL_JSON_PATH, "w+") as f:
        f.write(json.dumps(stringify_path_dict(our_to_original), indent=4))


def generate_patients_ours_mapping(
    patients_original_map: dict[str, dict], original_to_our: dict[Path, Path]
) -> dict[str, dict]:
    patients_ours = defaultdict(
        lambda: {
            "left": [],
            "right": [],
        }
    )

    for patient_id in patients_original_map:
        patient_path = Path(patient_id)
        patients_ours[patient_id]["left"] = list(
            map(
                lambda s: str(original_to_our[patient_path / s]),
                patients_original_map[patient_id]["left"],
            )
        )
        patients_ours[patient_id]["right"] = list(
            map(
                lambda s: str(original_to_our[patient_path / s]),
                patients_original_map[patient_id]["right"],
            )
        )

    return patients_ours


def save_patients_ours_mapping(patients_ours_map: dict[str, dict]):
    with open(PATIENTS_OURS_JSON_PATH, "w+") as f:
        f.write(json.dumps(patients_ours_map, indent=4))


def load_clean_labels(dataset: DatasetKind = DatasetKind.NUCLEAR_CATARACT):
    if dataset == DatasetKind.GABINET:
        return load_gabinet_labels()

    labels_df = pd.read_json(dataset.labels_path)
    labels_df = labels_df.dropna(subset=["choice"])
    labels_df = labels_df[
        labels_df["choice"].map(
            lambda x: x not in ["Zdjęcie nieczytelne", "Inne choroby"]
        )
    ]
    labels_df["choice"] = (
        labels_df["choice"].apply(lambda x: "Brak Zaćmy" if x == "Zdrowe" else x).copy()  # type: ignore
    )  # type: ignore
    labels_df["image"] = labels_df["image"].apply(lambda x: os.path.basename(x)).copy()  # type: ignore

    return labels_df


def gabinet_image_path(url: str) -> Path:
    """Turns a Label Studio local-files url into a path relative to the dataset root"""
    return Path(url.split(GABINET_URL_PREFIX, 1)[1])


def convert_bmp_to_png(path: Path) -> Path:
    """torchvision cannot decode BMP, so a PNG copy is kept next to the original"""
    if path.suffix.lower() != ".bmp":
        return path

    png_path = path.with_suffix(".png")
    full_png_path = DatasetKind.GABINET.root / png_path

    if not full_png_path.exists():
        Image.open(DatasetKind.GABINET.root / path).save(full_png_path)

    return png_path


def load_gabinet_labels() -> pd.DataFrame:
    """Flattens the per visit annotations into the same shape `load_clean_labels`
    returns, that is one row per image with `image`, `choice` and `label` (bboxes).

    Only the slit lamp images are taken, the other modalities are ignored. The
    diagnosis is given per visit, so every image of a visit shares its label.
    """
    with open(DatasetKind.GABINET.labels_path, encoding="utf-8") as f:
        visits = json.load(f)

    rows = []

    for visit in visits:
        choice = GABINET_DIAGNOSIS_MAPPING.get(visit.get("diagnosis"))

        # Diagnoses outside of the binary problem (e.g. "Inne choroby") are dropped,
        # the same way load_clean_labels drops them
        if choice is None:
            continue

        for idx, image in enumerate(visit["slitlamp"]):
            # Images the annotator did not mark as readable are dropped, the same
            # way "Zdjęcie nieczytelne" is dropped for nuclear-cataract-ours
            if visit.get(f"slitlamp_quality_{idx}") != GABINET_READABLE_QUALITY:
                continue

            path = convert_bmp_to_png(gabinet_image_path(image["url"]))
            rows.append(
                {
                    "image": path.name,
                    "path": str(path),
                    "patient_id": visit["patient_id"],
                    "choice": choice,
                    "label": visit.get(f"slitlamp_bbox_{idx}"),
                }
            )

    return pd.DataFrame(rows)


def load_gabinet_packs(labels_df: pd.DataFrame) -> list[list[Path]]:
    """One pack per visit, all of the images are of the same eye"""
    packs = defaultdict(list)

    for row in labels_df.itertuples():
        packs[row.patient_id].append(Path(row.path))

    return list(packs.values())


def get_label_of(labels_df: pd.DataFrame, image_path: str):
    return labels_df[labels_df["image"] == image_path]["choice"].item()


def get_bbox_of(labels_df: pd.DataFrame, image_path: str) -> list[list[float]]:
    """Returns a list of boxes in XYXY format.
    Each box is a list of four floats scaled from 0 to 1.
    xmin,ymin is top left and
    xmax,ymax is bottom right
    """
    row = labels_df[labels_df["image"] == image_path]
    if row.empty or "label" not in row.columns:
        return []
    label_field = row["label"].iloc[0]
    if not isinstance(label_field, list):
        return []
    bboxes = []
    for rect in label_field:
        x1 = rect["x"] / 100.0
        y1 = rect["y"] / 100.0
        x2 = (rect["x"] + rect["width"]) / 100.0
        y2 = (rect["y"] + rect["height"]) / 100.0
        bboxes.append([x1, y1, x2, y2])
    return bboxes


def load_unlabeled_packs(patients_ours_map: dict[str, dict]):
    datapacks = []

    for patient_id in patients_ours_map:
        patient = patients_ours_map[patient_id]
        left_eye = patient["left"]
        right_eye = patient["right"]

        if len(left_eye) != 0:
            datapacks.append(left_eye)
        if len(right_eye) != 0:
            datapacks.append(right_eye)

    return datapacks


def entropy(p_counts: list[int]) -> float:
    total = sum(p_counts)
    result = 0.0

    for count in p_counts:
        if count == 0:
            continue
        p = count / total
        result -= p * log(p)

    return result


def count_labeled_packs(labeled_packs: list[list[dict[str, str]]]) -> list[dict]:
    packs_counted = []

    for pack in labeled_packs:
        pack_labels = []
        pack_filtered = []

        for sample in pack:
            image_path = Path(sample["path"])
            try:
                image_label = sample["label"]
                pack_filtered.append(image_path)
                pack_labels.append(image_label)
            except Exception:
                pass

        cataract_label = "Zaćma"
        non_cataract_label = "Brak Zaćmy"
        cataracts_count = pack_labels.count(cataract_label)
        non_cataracts_count = pack_labels.count(non_cataract_label)

        packs_counted.append(
            {
                "pack": pack_filtered,
                "cataracts_count": cataracts_count,
                "non_cataracts_count": non_cataracts_count,
                "labels_entropy": entropy([cataracts_count, non_cataracts_count]),
            }
        )

    return packs_counted


def count_unlabeled_packs(datapacks: list[list[Path]], labels_df) -> list[dict]:
    labeled_packs = []

    for pack in datapacks:
        pack_filtered = []

        for image_path in pack:
            image_path = Path(image_path)
            try:
                image_label = get_label_of(labels_df, image_path.name)
                pack_filtered.append({"path": image_path, "label": image_label})
            except Exception:
                pass

        labeled_packs.append(pack_filtered)

    return count_labeled_packs(labeled_packs)


def generate_split_mapping(
    labels_df, subsets: list[list], subsets_names: list[str] | None = None
):
    result = {}

    if subsets_names is None:
        subsets_names = [f"{n}" for n in range(len(subsets))]

    for subset_name, subset in zip(subsets_names, subsets):
        result[subset_name] = []
        for pack in subset:
            result[subset_name].append(
                [
                    {"path": str(img), "label": get_label_of(labels_df, img.name)}
                    for img in pack["pack"]
                ]
            )

    return result


def save_split_mapping(split_mapping, dest_path):
    with open(dest_path, "w+") as f:
        f.write(json.dumps(split_mapping, indent=4, ensure_ascii=False))


def run_mode_initial(
    subsets_props: list[float],
    i_know_what_i_am_doing: bool = False,
    dataset: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
):
    # The gabinet split is derived from its labels alone and is deterministic,
    # so it can be regenerated at will
    if dataset == DatasetKind.GABINET:
        ensure_gabinet_deps()
        generate_split(subsets_props, load_gabinet_packs(load_gabinet_labels()), dataset)
        return

    assert i_know_what_i_am_doing

    DATA_BASE_PATH.mkdir(exist_ok=True)

    if not ORIGINAL_DATASET_PATH.exists():
        zip_cache_path = download_files([ORIGINAL_DATASET_ZIP_NAME])[0]
        with zipfile.ZipFile(zip_cache_path, "r") as zip_ref:
            zip_ref.extractall(DATA_BASE_PATH)

    if not OURS_DATASET_PATH.exists():
        zip_cache_path = download_files([OURS_DATASET_ZIP_NAME])[0]
        with zipfile.ZipFile(zip_cache_path, "r") as zip_ref:
            zip_ref.extractall(DATA_BASE_PATH)
        shutil.move(
            DATA_BASE_PATH / "Nuclear_Cataract_2025_12_21" / "Nuclear Cataract",
            DATA_BASE_PATH,
        )
        (DATA_BASE_PATH / "Nuclear_Cataract_2025_12_21").rmdir()
        shutil.move(DATA_BASE_PATH / "Nuclear Cataract", OURS_DATASET_PATH)

    if not GENERATED_DIR_PATH.exists():
        GENERATED_DIR_PATH.mkdir()

    patients_original_map = generate_patients_original_mapping()
    save_patients_original_mapping(patients_original_map)

    our_images_bytes = load_our_images_to_memory()
    original_to_our, our_to_original = generate_our_to_original_mappings(
        our_images_bytes, patients_original_map
    )
    save_our_to_original_mappings(original_to_our, our_to_original)
    patients_ours_map = generate_patients_ours_mapping(
        patients_original_map, original_to_our
    )
    save_patients_ours_mapping(patients_ours_map)
    generate_split(subsets_props, load_unlabeled_packs(patients_ours_map), dataset)


def split_packs(
    counted_packs: list[dict], subsets_props: list[float]
) -> tuple[SubsetSplitter | None, list[list]]:
    """SubsetSplitter cannot express a degenerate split, so the subsets with a
    zero proportion are kept out of it and stay empty.
    """
    non_zero = [i for i, prop in enumerate(subsets_props) if prop > 0]
    subsets: list[list] = [[] for _ in subsets_props]

    if len(non_zero) == 1:
        subsets[non_zero[0]] = counted_packs
        return None, subsets

    ss = SubsetSplitter(counted_packs, [subsets_props[i] for i in non_zero])

    for subset_idx, subset in zip(non_zero, ss.subsets):
        subsets[subset_idx] = subset

    return ss, subsets


def generate_split(
    subsets_props: list[float],
    datapacks: list[list[Path]],
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
):
    """Splits the packs into trainval/test and writes the dataset's split.json"""
    labels_df = load_clean_labels(dataset_kind)
    datapacks_counted = count_unlabeled_packs(datapacks, labels_df)

    ss, subsets = split_packs(datapacks_counted, subsets_props)
    sm = generate_split_mapping(
        labels_df,
        [subsets[0] + subsets[1], subsets[2]],
        ["trainvalSet", "testSet"],
    )
    save_split_mapping(sm, dataset_kind.split_json_path)

    if ss:
        ss.print_stats()


def ensure_gabinet_deps():
    if not DatasetKind.GABINET.root.exists():
        DATA_BASE_PATH.mkdir(exist_ok=True)

        zip_cache_path = download_files([GABINET_ZIP_NAME])[0]
        password = os.environ.get(GABINET_ZIP_PASSWORD_ENV) or getpass(
            "Archive password: "
        )

        # The archive is AES encrypted, which zipfile cannot read
        with pyzipper.AESZipFile(zip_cache_path, "r") as zip_ref:
            zip_ref.extractall(DATA_BASE_PATH, pwd=password.encode())


def ensure_runtime_deps(dataset: DatasetKind = DatasetKind.NUCLEAR_CATARACT):
    if dataset == DatasetKind.GABINET:
        ensure_gabinet_deps()
        return

    if not OURS_DATASET_PATH.exists() or not GENERATED_DIR_PATH.exists():
        if DATA_BASE_PATH.exists():
            data_files = list(DATA_BASE_PATH.iterdir())

            if len(data_files) > 1 or (
                len(data_files) == 1 and data_files[0] != ".gitkeep"
            ):
                raise Exception("data directory is not sane")

        DATA_BASE_PATH.mkdir(exist_ok=True)

        zip_cache_path = download_files([PREPARED_DATASET_ZIP_NAME])[0]
        with zipfile.ZipFile(zip_cache_path, "r") as zip_ref:
            zip_ref.extractall(DATA_BASE_PATH / "Nuclear_Cataract")

        shutil.move(DATA_BASE_PATH / "Nuclear_Cataract" / "generated", DATA_BASE_PATH)
        shutil.move(DATA_BASE_PATH / "Nuclear_Cataract", OURS_DATASET_PATH)


def load_test_set(
    should_flatten_packs: bool = False,
    hard_policy: HardPolicy = HardPolicy.PASSTHROUGH,
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
):
    ensure_runtime_deps(dataset_kind)

    with open(dataset_kind.split_json_path) as split:
        split_json = json.load(split)

    ts = split_json["testSet"]
    ts = hard_policy.apply(ts)

    if should_flatten_packs:
        ts = flatten_packs(ts)

    return ts


def split_trainval_set(
    subsets_props: list[float],
    subsets_names: list[str] | None = None,
    should_flatten_packs: bool = False,
    hard_policy: HardPolicy = HardPolicy.PASSTHROUGH,
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
) -> tuple[SubsetSplitter, dict]:
    with open(dataset_kind.split_json_path) as split:
        split_mapping = json.load(split)

    trainval_split = split_mapping["trainvalSet"]

    if not trainval_split:
        raise RuntimeError(
            f"trainvalSet of {dataset_kind} is empty, regenerate the split with non zero "
            f"train and val proportions: -dataset {dataset_kind} -initial <train> <val> <test>"
        )

    trainval_split = hard_policy.apply(trainval_split)

    labels_df = load_clean_labels(dataset_kind)
    counted_packs = count_labeled_packs(trainval_split)

    ss = SubsetSplitter(counted_packs, subsets_props)
    sm = generate_split_mapping(labels_df, ss.subsets, subsets_names)

    if should_flatten_packs:
        for key in sm:
            sm[key] = flatten_packs(sm[key])

    return ss, sm


def flatten_packs(packs: list[list[dict]]):
    result: list[dict[str, str]] = []
    for pack in packs:
        result.extend(pack)
    return result


def split_mode_helper(
    subsets_props: list[float],
    subsets_names: list[str] | None = None,
    save_mapping_path: Path | None = None,
    print_stats: bool = False,
    should_flatten_packs: bool = False,
    hard_policy: HardPolicy = HardPolicy.PASSTHROUGH,
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
):
    ensure_runtime_deps(dataset_kind)
    ss, sm = split_trainval_set(
        subsets_props, subsets_names, should_flatten_packs, hard_policy, dataset_kind
    )

    if save_mapping_path:
        save_split_mapping(sm, save_mapping_path)

    if print_stats:
        ss.print_stats()

    return sm


def run_mode_kfoldcv(
    k: int,
    save_mapping: bool = False,
    print_stats: bool = False,
    should_flatten_packs: bool = False,
    hard_policy: HardPolicy = HardPolicy.PASSTHROUGH,
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
) -> dict:
    save_mapping_path = KFOLDCV_JSON_PATH if save_mapping else None
    return split_mode_helper(
        subsets_props=[float(1) / k for _ in range(k)],
        save_mapping_path=save_mapping_path,
        print_stats=print_stats,
        should_flatten_packs=should_flatten_packs,
        hard_policy=hard_policy,
        dataset_kind=dataset_kind,
    )


def run_mode_deps():
    download_deps()
    download_dataset()


def run_mode_zipgen():
    if (
        not ORIGINAL_DATASET_PATH.exists()
        or not OURS_DATASET_PATH.exists()
        or not GENERATED_DIR_PATH.exists()
    ):
        raise RuntimeError("dataset is not ready for zip generation")

    ZIPGEN_TMP_DIR.mkdir()
    shutil.copytree(GENERATED_DIR_PATH, ZIPGEN_TMP_DIR / GENERATED_DIR_PATH.name)

    for f in OURS_DATASET_PATH.iterdir():
        if f.name not in [KFOLDCV_JSON_PATH.name, TRAINVAL_JSON_PATH.name]:
            if f.is_dir():
                shutil.copytree(f, ZIPGEN_TMP_DIR / f.name)
            else:
                shutil.copy(f, ZIPGEN_TMP_DIR)

    OUTPUT_DIR.mkdir(exist_ok=True)
    shutil.make_archive(str(ZIPGEN_OUTPUT_ZIP_PATH_WITHOUT_EXT), "zip", ZIPGEN_TMP_DIR)
    shutil.rmtree(ZIPGEN_TMP_DIR)


def run_mode_trainval(
    train_prop: float,
    val_prop: float,
    save_mapping: bool = False,
    print_stats: bool = False,
    should_flatten_packs: bool = False,
    hard_policy: HardPolicy = HardPolicy.PASSTHROUGH,
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
) -> dict:
    save_mapping_path = TRAINVAL_JSON_PATH if save_mapping else None
    return split_mode_helper(
        [train_prop, val_prop],
        ["train", "val"],
        save_mapping_path,
        print_stats,
        should_flatten_packs,
        hard_policy,
        dataset_kind,
    )


def get_class_mapping_from_elements(elements: list, are_packs: bool = True):
    classes = set()

    if not are_packs:
        elements = [elements]

    for pack in elements:
        for sample in pack:
            classes.add(sample["label"])

    classes = sorted(classes)
    class_mapping: dict[str, int] = {}

    for cls_idx, cls in enumerate(classes):
        class_mapping[cls] = cls_idx

    return class_mapping


def run_mode_classes(
    save_mapping: bool,
    dataset_kind: DatasetKind = DatasetKind.NUCLEAR_CATARACT,
) -> dict:
    with open(dataset_kind.split_json_path) as split:
        split_json = json.load(split)

    all_packs = []

    for subset_name in split_json:
        all_packs.extend(split_json[subset_name])

    class_mapping = get_class_mapping_from_elements(all_packs)

    if save_mapping:
        with open(CLASSES_JSON_PATH, "w") as f:
            f.write(json.dumps(class_mapping, indent=4, ensure_ascii=False))

    return class_mapping


def download_deps():
    names = ["Nuclear_Cataract_2025_12_21.zip", "Nuclear_Cataract_Original.zip"]
    download_files(names)


def download_dataset():
    names = ["Nuclear_Cataract_2026_04_30.zip"]
    download_files(names)


def download_files(names: list[str]) -> list[Path]:
    if not CACHE_PATH.exists():
        CACHE_PATH.mkdir()

    deps_cache_paths = [CACHE_PATH / dep for dep in names]
    needs_download = [(d, p) for d, p in zip(names, deps_cache_paths) if not p.exists()]

    if len(needs_download) > 0:
        print("Some deps needs to be downloaded, please authenticate")
        username = input("Username: ")
        password = getpass("Password: ")
        basic = HTTPBasicAuth(username, password)

        for dep_name, dep_cache_path in needs_download:
            print(f"Downloading {dep_name} -> {dep_cache_path}")
            r = requests.get(
                f"{FILESERVER_URL}/{dep_name}",
                auth=basic,
                stream=True,
            )
            with open(dep_cache_path, "wb") as dest_file:
                with tqdm.tqdm(unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                    for chunk in r.iter_content(chunk_size=1024):
                        dest_file.write(chunk)
                        pbar.update(1024)
            print(f"\nDownloaded {dep_name}")

    return deps_cache_paths


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i-know-what-i-am-doing", action="store_true")
    parser.add_argument(
        "-dataset",
        type=DatasetKind,
        choices=list(DatasetKind),
        default=DatasetKind.NUCLEAR_CATARACT,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "-initial", nargs=3, metavar=("train", "val", "test"), type=float
    )
    group.add_argument("-kfoldcv", type=int, metavar="k")
    group.add_argument("-deps", type=bool)
    group.add_argument("-zipgen", type=bool)
    group.add_argument("-trainval", nargs=2, metavar=("train", "val"), type=float)
    group.add_argument("-classes", type=bool)

    args = parser.parse_args()

    if args.initial is not None:
        run_mode_initial(args.initial, args.i_know_what_i_am_doing, args.dataset)
    elif args.kfoldcv is not None:
        run_mode_kfoldcv(
            args.kfoldcv,
            save_mapping=True,
            print_stats=True,
            should_flatten_packs=True,
            dataset_kind=args.dataset,
        )
    elif args.deps is not None:
        run_mode_deps()
    elif args.zipgen is not None:
        run_mode_zipgen()
    elif args.trainval is not None:
        run_mode_trainval(
            args.trainval[0],
            args.trainval[1],
            save_mapping=True,
            print_stats=True,
            should_flatten_packs=True,
            dataset_kind=args.dataset,
        )
    elif args.classes is not None:
        run_mode_classes(
            save_mapping=True,
            dataset_kind=args.dataset,
        )


if __name__ == "__main__":
    main()
