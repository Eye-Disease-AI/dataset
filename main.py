import os
import json
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import time

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_BASE_PATH = SCRIPT_DIR / "data"
ORIGINAL_DATASET_PATH = DATA_BASE_PATH / "nuclear-cataract-original"
OURS_DATASET_PATH = DATA_BASE_PATH / "nuclear-cataract-ours"
GENERATED_DIR_PATH = DATA_BASE_PATH / "generated"
PATIENTS_JSON_PATH = GENERATED_DIR_PATH / "patients_original.json"

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

    for root, dirs, files in os.walk(dir_path):
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

def save_patients_map(patients_map: dict[str, dict]):
    patients_images_json = json.dumps(patients_map, indent=4)
    with open(PATIENTS_JSON_PATH, "w+") as f:
        f.write(patients_images_json)

def load_our_images_to_memory(patients_map: dict[str, dict]) -> dict[Path, bytes]:
    our_images_dir = OURS_DATASET_PATH / "images"
    original_images_paths = []

    for patient_id in patients_map.keys():
        for image_path in (patients_map[patient_id]["left"] + patients_map[patient_id]["right"]):
            original_images_paths.append(ORIGINAL_DATASET_PATH / patient_id / image_path)

    our_images_paths = list(our_images_dir.iterdir())
    our_images_bytes = {}

    for our_image_path in tqdm.tqdm(our_images_paths, desc="loading our images"):
        with open(our_image_path, "rb") as f:
            our_image_bytes = f.read()
        our_images_bytes[our_image_path] = our_image_bytes
    
    return our_images_bytes

if __name__ == "__main__":
    if not GENERATED_DIR_PATH.exists():
        GENERATED_DIR_PATH.mkdir()

    patients_map = generate_patients_original_mapping()
    save_patients_map(patients_map)
    
    our_images_bytes = load_our_images_to_memory(patients_map)