import os

# ROOT
PROJECT_DIR = '/path/to/project/directory'

# RESULTS
RESULTS_DIR = f'{PROJECT_DIR}/results/'

# MODELS
EMBEDDING_MODEL_DIR = f'{PROJECT_DIR}/saved_models/embedding_models/'
SAVED_MODELS_DIR = f'{PROJECT_DIR}/saved_models/'

# DATASETS
EMEDDED_DATASETS_DIR = f'{PROJECT_DIR}/datasets/embedded_datasets/'
PROCESSED_DATASETS_DIR = f'{PROJECT_DIR}/datasets/'
# CUB Constants
# CUB data is downloaded from the CBM release.
# Dataset: https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/
# Metadata and splits: https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683
CUB_DATA_DIR = f'{PROJECT_DIR}/datasets_raw/CUB_200_2011'
CUB_PROCESSED_DIR = f'{PROJECT_DIR}/datasets_raw/class_attr_data_10'
# Derm data constants
# Derm7pt is obtained from : https://derm.cs.sfu.ca/Welcome.html
DERM7_FOLDER = f'{PROJECT_DIR}/datasets_raw/derm7pt/release_v0'
DERM7_META = os.path.join(DERM7_FOLDER, "meta", "meta.csv")
DERM7_TRAIN_IDX = os.path.join(DERM7_FOLDER, "meta", "train_indexes.csv")
DERM7_VAL_IDX = os.path.join(DERM7_FOLDER, "meta", "valid_indexes.csv")
# Ham10000 can be obtained from : https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
# HAM10K_DATA_DIR = "/path/to/broden/"
# # BRODEN concept bank
# BRODEN_CONCEPTS = "/path/to/broden/"