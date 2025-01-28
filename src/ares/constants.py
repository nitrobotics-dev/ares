import os
from collections import defaultdict

ARES_DATA_DIR = "/workspaces/ares/data"
ARES_OXE_DIR = os.path.join(ARES_DATA_DIR, "oxe")
ARES_VIDEO_DIR = os.path.join(ARES_DATA_DIR, "videos")

# using oxe-downloader
# oxe-download --dataset "name" --path $ARES_OXE_DIR!!!
DATASET_NAMES = [
    # {
    #     "dataset_filename": "ucsd_kitchen_dataset_converted_externally_to_rlds",
    #     "dataset_formalname": "UCSD Kitchen",
    # },
    # {
    #     "dataset_filename": "cmu_franka_exploration_dataset_converted_externally_to_rlds",
    #     "dataset_formalname": "CMU Franka Exploration",
    # },
    # {
    #     "dataset_filename": "berkeley_fanuc_manipulation",
    #     "dataset_formalname": "Berkeley Fanuc Manipulation",
    # },
    # {
    #     "dataset_filename": "cmu_stretch",
    #     "dataset_formalname": "CMU Stretch",
    # },
    # {"dataset_filename": "cmu_play_fusion", "dataset_formalname": "CMU Play Fusion"},
    # {
    #     "dataset_filename": "jaco_play",
    #     "dataset_formalname": "USC Jaco Play",
    # },
    # {
    #     "dataset_filename": "dlr_edan_shared_control_converted_externally_to_rlds",
    #     "dataset_formalname": "DLR Wheelchair Shared Control",
    # },
    # {
    #     "dataset_filename": "imperialcollege_sawyer_wrist_cam",
    #     "dataset_formalname": "Imperial Wrist Cam",
    # },
    # {
    #     "dataset_filename": "tokyo_u_lsmo_converted_externally_to_rlds",
    #     "dataset_formalname": "LSMO Dataset",
    # },
    # {
    #     "dataset_filename": "nyu_rot_dataset_converted_externally_to_rlds",
    #     "dataset_formalname": "NYU ROT",
    # },
    # {
    #     "dataset_filename": "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
    #     "dataset_formalname": "UCSD Pick Place",
    # },
    # {
    #     "dataset_filename": "asu_table_top_converted_externally_to_rlds",
    #     "dataset_formalname": "ASU TableTop Manipulation",
    # },
    # {
    #     "dataset_filename": "viola",
    #     "dataset_formalname": "Austin VIOLA",
    # },
    # {
    #     "dataset_filename": "kaist_nonprehensile_converted_externally_to_rlds",
    #     "dataset_formalname": "KAIST Nonprehensile Objects",
    # },
    # {
    #     "dataset_filename": "berkeley_mvp_converted_externally_to_rlds",
    #     "dataset_formalname": "Berkeley MVP Data",
    # },
    # Saytap does not have pixel data, so we exclude it
    # {
    #     "dataset_filename": "utokyo_saytap_converted_externally_to_rlds",
    #     "dataset_formalname": "Saytap",
    # },
]

DATASET_KEY_TO_DATASET_INFO = defaultdict(dict)
keys = ["dataset_filename", "dataset_formalname"]
for dataset_info in DATASET_NAMES:
    for key in keys:
        DATASET_KEY_TO_DATASET_INFO[key][dataset_info[key]] = dataset_info


def get_dataset_info_by_key(key_type: str, key: str) -> dict:
    if key_type not in DATASET_KEY_TO_DATASET_INFO:
        raise ValueError(f"Invalid key type: {key_type}")
    if key not in DATASET_KEY_TO_DATASET_INFO[key_type]:
        raise ValueError(f"Invalid key: {key}")
    return DATASET_KEY_TO_DATASET_INFO[key_type][key]


# for ingestion operations, we're loading large amounts of data into memory at once.
# this is a hard limit on the number of rollouts/requests to avoid memory issues.
OUTER_BATCH_SIZE = 20

# for annotation operations, the objects in memory are smaller (eg no point clouds),
# so we can load more into memory at once.
ANNOTATION_OUTER_BATCH_SIZE = 100
