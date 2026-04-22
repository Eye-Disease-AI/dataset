import os
import json
from pathlib import Path
import tqdm
from collections import defaultdict
import json
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_BASE_PATH = SCRIPT_DIR / "data"
ORIGINAL_DATASET_PATH = DATA_BASE_PATH / "nuclear-cataract-original"
OURS_DATASET_PATH = DATA_BASE_PATH / "nuclear-cataract-ours"
OUR_IMAGES_DIR = OURS_DATASET_PATH / "images"
GENERATED_DIR_PATH = DATA_BASE_PATH / "generated"
PATIENTS_ORIGINAL_JSON_PATH = GENERATED_DIR_PATH / "patients_original.json"
OUR_TO_ORIGINAL_JSON_PATH = GENERATED_DIR_PATH / "our_to_original.json"
ORIGINAL_TO_OUR_JSON_PATH = GENERATED_DIR_PATH / "original_to_our.json"
PATIENTS_OURS_JSON_PATH = GENERATED_DIR_PATH / "patients_ours.json"

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
    original_patients_dirs = list(filter(original_dataset_filter, original_patients_dirs))

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

        patient_dir_name_rel_str = str(patient_dir_name.relative_to(ORIGINAL_DATASET_PATH))
        patients_map[patient_dir_name_rel_str] = {
            "left": list(map(lambda p: str(p.relative_to(patient_dir_path)), left_eyes_paths)),
            "right": list(map(lambda p: str(p.relative_to(patient_dir_path)), right_eyes_paths)),
        }

    return patients_map

def save_patients_original_mapping(patients_original_map: dict[str, dict]):
    patients_images_json = json.dumps(patients_original_map, indent=4)
    with open(PATIENTS_ORIGINAL_JSON_PATH, "w+") as f:
        f.write(patients_images_json)

def get_original_images_paths(patients_original_map: dict[str, dict]) -> list[Path]:
    original_images_paths = []

    for patient_id in patients_original_map.keys():
        for image_path in (patients_original_map[patient_id]["left"] + patients_original_map[patient_id]["right"]):
            original_images_paths.append(ORIGINAL_DATASET_PATH / patient_id / image_path)
    
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

def generate_our_to_original_mappings(our_images_bytes: dict[Path, bytes], patients_original_map: dict[str, dict]) -> tuple[dict[Path, Path], dict[Path, Path]]:
    original_images_paths = get_original_images_paths(patients_original_map)
    original_to_our = {}
    our_to_original = {}

    our_images_paths = list(OUR_IMAGES_DIR.iterdir())

    for original_image_path in tqdm.tqdm(original_images_paths, desc="comparing images"):
        with open(original_image_path, "rb") as f:
            original_image_bytes = f.read()

        for our_image_path in our_images_paths:
            our_image_bytes = our_images_bytes[our_image_path]
            are_identical = original_image_bytes == our_image_bytes

            if are_identical:
                original_to_our[original_image_path.relative_to(ORIGINAL_DATASET_PATH)] = our_image_path.relative_to(OURS_DATASET_PATH)
                our_to_original[our_image_path.relative_to(OURS_DATASET_PATH)] = original_image_path.relative_to(ORIGINAL_DATASET_PATH)
                our_images_paths.remove(our_image_path)
                break

    return original_to_our, our_to_original

def save_our_to_original_mappings(original_to_our: dict[Path, Path], our_to_original: dict[Path, Path]):
    with open(ORIGINAL_TO_OUR_JSON_PATH, "w+") as f:
        f.write(json.dumps(stringify_path_dict(original_to_our), indent=4))

    with open(OUR_TO_ORIGINAL_JSON_PATH, "w+") as f:
        f.write(json.dumps(stringify_path_dict(our_to_original), indent=4))

def generate_patients_ours_mapping(patients_original_map: dict[str, dict], original_to_our: dict[Path, Path]) -> dict[str, dict]:
    patients_ours = defaultdict(lambda: {
        "left": [],
        "right": [],
    })

    for patient_id in patients_original_map:
        patient_path = Path(patient_id)
        patients_ours[patient_id]["left"] = list(map(lambda s: str(original_to_our[patient_path / s]), patients_original_map[patient_id]["left"]))
        patients_ours[patient_id]["right"] = list(map(lambda s: str(original_to_our[patient_path / s]), patients_original_map[patient_id]["right"]))

    return patients_ours

def save_patients_ours_mapping(patients_ours_map: dict[str, dict]):
    with open(PATIENTS_OURS_JSON_PATH, "w+") as f:
        f.write(json.dumps(patients_ours_map, indent=4))

def load_clean_labels():
    labels_df = pd.read_json(OURS_DATASET_PATH / "labels.json")
    labels_df = labels_df.dropna(subset=["choice"])
    labels_df = labels_df[labels_df["choice"].map(lambda x: x not in ["Zdjęcie nieczytelne", "Inne choroby"])]
    labels_df["choice"] = labels_df["choice"].apply(lambda x: "Brak Zaćmy" if x == "Zdrowe" else x).copy() # type: ignore
    labels_df["image"] = labels_df["image"].apply(lambda x: os.path.basename(x)).copy() # type: ignore

    return labels_df

def get_label_of(labels_df: pd.DataFrame, image_path: str):
    return labels_df[labels_df["image"] == image_path]["choice"].item()

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

def count_packs(datapacks: list[list[Path]], labels_df) -> list[dict]:
    packs_counted = []

    print(labels_df)

    for pack in datapacks:
        pack_labels = []
        pack_filtered = []

        for image_path in pack:
            image_path = Path(image_path)
            try:
                image_label = get_label_of(labels_df, image_path.name)
                pack_filtered.append(image_path)
                pack_labels.append(image_label)
            except:
                pass

        cataract_label = "Zaćma"
        non_cataract_label = "Brak Zaćmy"
        cataracts_count = pack_labels.count(cataract_label)
        non_cataracts_count = pack_labels.count(non_cataract_label)

        packs_counted.append({
            "pack": pack_filtered,
            "cataracts_count": cataracts_count,
            "non_cataracts_count": non_cataracts_count,
        })
    
    return packs_counted

if __name__ == "__main__":
    if not GENERATED_DIR_PATH.exists():
        GENERATED_DIR_PATH.mkdir()

    patients_original_map = generate_patients_original_mapping()
    save_patients_original_mapping(patients_original_map)

    our_images_bytes = load_our_images_to_memory()
    original_to_our, our_to_original = generate_our_to_original_mappings(our_images_bytes, patients_original_map)
    save_our_to_original_mappings(original_to_our, our_to_original)
    patients_ours_map = generate_patients_ours_mapping(patients_original_map, original_to_our)
    save_patients_ours_mapping(patients_ours_map)
    labels_df = load_clean_labels()
    datapacks = load_unlabeled_packs(patients_ours_map)
    datapacks_counted = count_packs(datapacks, labels_df)
    from pprint import pprint
    pprint(datapacks_counted[3])