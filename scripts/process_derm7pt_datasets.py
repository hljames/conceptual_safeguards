"""
Process Derm7pt datasets
"""
import argparse
import dill
import numpy as np
import os
import pandas as pd
import random

from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from sklearn.preprocessing import LabelEncoder

from constants import DERM7_META, DERM7_TRAIN_IDX, DERM7_VAL_IDX, PROCESSED_DATASETS_DIR, DERM7_FOLDER
from data.image_data import ImageCBMDataset, augment_training_dataset
from data.data import downsample_to_balance_y_classes, majority_denoise_dataset
from utils import str2bool


def config():
    default_config = {
        'random_seed': 1,
        'reduce_to_binary': 'False',
        'basal_cell': 'False',
        'melanoma': 'True',
        'balance_y_classes': 'False',
        'drop_basal_cell': 'False',
        'majority_denoising': 'False',
        'combine_train_val': 'False',
        'augmentation_factor': 1.0,
        'n_concepts': -1,
        'suffix': 'v2_supp_concepts',
    }
    return default_config


# what if we drop basal cell carcinoma?
CANCEROUS_CLASSES = ['basal cell carcinoma', 'melanoma (in situ)', 'melanoma (less than 0.76 mm)',
                     'melanoma (0.76 to 1.5 mm)', 'melanoma (more than 1.5 mm)', 'melanoma metastasis']


def process_derm7pt_datasets(meta_file_path, train_idx_file_path, val_idx_file_path, n_concepts=-1,
                             reduce_to_binary=False, combine_train_val=False, augmentation_factor=1.0,
                             basal_cell=False, melanoma=False, balance_y_classes=False, drop_basal_cell=False,
                             majority_denoising=False):
    df = pd.read_csv(meta_file_path)
    train_indexes = list(pd.read_csv(train_idx_file_path)['indexes'])
    val_indexes_all = list(pd.read_csv(val_idx_file_path)['indexes'])
    # randomly split val_indexes into val and test
    np.random.shuffle(val_indexes_all)
    val_indexes = val_indexes_all[:len(val_indexes_all) // 2]
    test_indexes = val_indexes_all[len(val_indexes_all) // 2:]
    assert len(set(train_indexes).intersection(set(val_indexes))) == 0
    assert len(set(train_indexes).intersection(set(test_indexes))) == 0
    assert len(set(val_indexes).intersection(set(test_indexes))) == 0
    if combine_train_val:
        train_indexes.extend(val_indexes)
    # pigment network
    df["TypicalPigmentNetwork"] = df.apply(
        lambda row: {"absent": 0, "typical": 1, "atypical": 0}[row["pigment_network"]], axis=1)
    df["AtypicalPigmentNetwork"] = df.apply(
        lambda row: {"absent": 0, "typical": 0, "atypical": 1}[row["pigment_network"]], axis=1)
    df["AbsentPigmentNetwork"] = df.apply(
        lambda row: {"absent": 1, "typical": 0, "atypical": 0}[row["pigment_network"]], axis=1)

    # regression structures
    df["RegressionStructures"] = df.apply(
        lambda row: {"absent": 0, "white areas": 1, "blue areas": 1, "combinations": 1}[
            row["regression_structures"]], axis=1)
    df["WhiteAreaRegressionStructures"] = df.apply(
        lambda row: {"absent": 0, "white areas": 1, "blue areas": 0, "combinations": 0}[
            row["regression_structures"]], axis=1)
    df["BlueAreaRegressionStructures"] = df.apply(
        lambda row: {"absent": 0, "white areas": 0, "blue areas": 1, "combinations": 0}[
            row["regression_structures"]], axis=1)
    df["CombinationRegressionStructures"] = df.apply(
        lambda row: {"absent": 0, "white areas": 0, "blue areas": 0, "combinations": 1}[
            row["regression_structures"]], axis=1)

    df["BWV"] = df.apply(lambda row: {"absent": 0, "present": 1}[row["blue_whitish_veil"]], axis=1)

    # regular vascular structures
    df["RegularVascularStructures"] = df.apply(lambda row: {"absent": 0,
                                                            "regular": 1, "arborizing": 1, "comma": 1, "hairpin": 1,
                                                            "within regression": 1, "wreath": 1,
                                                            "dotted/irregular": 0, "dotted": 0,
                                                            "linear irregular": 0, }[row["vascular_structures"]],
                                               axis=1)
    df["IrregularVascularStructures"] = df.apply(lambda row: {"absent": 0,
                                                              "regular": 0, "arborizing": 0, "comma": 0, "hairpin": 0,
                                                              "within regression": 0, "wreath": 0,
                                                              "dotted/irregular": 1, "dotted": 1,
                                                              "linear irregular": 1, }[row["vascular_structures"]],
                                                 axis=1)
    df["AbsentVascularStructures"] = df.apply(lambda row: {"absent": 1,
                                                           "regular": 0, "arborizing": 0, "comma": 0, "hairpin": 0,
                                                           "within regression": 0, "wreath": 0,
                                                           "dotted/irregular": 0, "dotted": 0,
                                                           "linear irregular": 0, }[row["vascular_structures"]],
                                              axis=1)
    df["DottedVascularStructures"] = df.apply(lambda row: {"absent": 0,
                                                           "regular": 0, "arborizing": 0, "comma": 0, "hairpin": 0,
                                                           "within regression": 0, "wreath": 0,
                                                           "dotted/irregular": 1, "dotted": 1,
                                                           "linear irregular": 0, }[row["vascular_structures"]],
                                              axis=1)
    df["WithinRegressionVascularStructures"] = df.apply(lambda row: {"absent": 0,
                                                                     "regular": 0, "arborizing": 0, "comma": 0,
                                                                     "hairpin": 0,
                                                                     "within regression": 1, "wreath": 0,
                                                                     "dotted/irregular": 0, "dotted": 0,
                                                                     "linear irregular": 0, }[
        row["vascular_structures"]],
                                                        axis=1)
    df["ArborizingVascularStructures"] = df.apply(lambda row: {"absent": 0,
                                                               "regular": 0, "arborizing": 1, "comma": 0, "hairpin": 0,
                                                               "within regression": 0, "wreath": 0,
                                                               "dotted/irregular": 0, "dotted": 0,
                                                               "linear irregular": 0, }[row["vascular_structures"]],
                                                  axis=1)
    # streaks
    df["RegularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": 0}[row["streaks"]],
                                    axis=1)
    df["IrregularStreaks"] = df.apply(lambda row: {"absent": 0, "regular": 0, "irregular": 1}[row["streaks"]],
                                      axis=1)
    df["AbsentStreaks"] = df.apply(lambda row: {"absent": 1, "regular": 0, "irregular": 0}[row["streaks"]],
                                   axis=1)
    # dots and globules
    df["RegularDG"] = df.apply(lambda row: {"absent": 0, "regular": 1, "irregular": 0}[row["dots_and_globules"]],
                               axis=1)
    df["IrregularDG"] = df.apply(lambda row: {"absent": 0, "regular": 0, "irregular": 1}[row["dots_and_globules"]],
                                 axis=1)
    df["AbsentDG"] = df.apply(lambda row: {"absent": 1, "regular": 0, "irregular": 0}[row["dots_and_globules"]],
                              axis=1)
    # df = df.iloc[train_indexes+val_indexes+test_indexes]
    # set indices to

    concepts = ["TypicalPigmentNetwork", "AtypicalPigmentNetwork", "AbsentPigmentNetwork",
                "RegressionStructures", "WhiteAreaRegressionStructures", "BlueAreaRegressionStructures",
                "CombinationRegressionStructures",
                "BWV",
                "RegularVascularStructures", "IrregularVascularStructures", "AbsentVascularStructures",
                "RegularStreaks", "IrregularStreaks", "AbsentStreaks",
                "RegularDG", "IrregularDG", "AbsentDG"]

    X = df["derm"].values
    C = df[concepts].values
    y = df["diagnosis"].values
    # convert y values to integer values
    if reduce_to_binary:
        y = np.array([1 if y_i in CANCEROUS_CLASSES else 0 for y_i in y])
        print(f"Reduced to binary, {np.sum(y)} cancerous, {len(y) - np.sum(y)} non-cancerous")
    elif basal_cell:
        y = np.array([1 if y_i == "basal cell carcinoma" else 0 for y_i in y])
        print(f"Reduced to basal cell carcinoma, {np.sum(y)} pos, {len(y) - np.sum(y)} neg")
    elif melanoma:
        y = np.array([1 if "melanoma" in y_i else 0 for y_i in y])
        print(f"Reduced to melanoma, {np.sum(y)} pos, {len(y) - np.sum(y)} neg")
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    base_dir = os.path.join(DERM7_FOLDER, "images")
    if drop_basal_cell:
        # find indices with basal cell carcinoma
        basal_cell_indexes = df[df["diagnosis"] == "basal cell carcinoma"].index
        print(f'Removing {len(basal_cell_indexes)} basal cell carcinoma images')
        # remove these indices from training, validation and test indexes
        train_indexes = [i for i in train_indexes if i not in basal_cell_indexes]
        val_indexes = [i for i in val_indexes if i not in basal_cell_indexes]
        test_indexes = [i for i in test_indexes if i not in basal_cell_indexes]

    train_dataset = ImageCBMDataset(X=X[train_indexes], C=C[train_indexes], y=y[train_indexes], base_dir=base_dir,
                                    concept_names=concepts)
    if majority_denoising:
        # denoise training dataset by flipping concepts to the majority for the class
        print('denoising training dataset with majority voting')
        train_dataset = majority_denoise_dataset(train_dataset)
    if augmentation_factor > 1.0:
        print(f'Augmenting training dataset, size before augmentation: {len(train_dataset)}')
        train_dataset = augment_training_dataset(train_dataset, augmentation_factor)
        print(f'Augmented training dataset size: {len(train_dataset)}')
    val_dataset = ImageCBMDataset(X=X[val_indexes], C=C[val_indexes], y=y[val_indexes], base_dir=base_dir,
                                  concept_names=concepts)
    test_dataset = ImageCBMDataset(X=X[test_indexes], C=C[test_indexes], y=y[test_indexes], base_dir=base_dir,
                                   concept_names=concepts)
    if balance_y_classes:
        print('downsampling to balance y classes')
        print(f'before downsampling: train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')
        train_dataset = downsample_to_balance_y_classes(train_dataset)
        val_dataset = downsample_to_balance_y_classes(val_dataset)
        test_dataset = downsample_to_balance_y_classes(test_dataset)
        print(f'after downsampling: train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    config = config()
    parser = argparse.ArgumentParser(description='Process Derm7pt datasets')
    parser.add_argument('--random-seed', type=int, default=config['random_seed'],
                        help=f'Random seed (default {config["random_seed"]})'),
    parser.add_argument('--n-concepts', type=int, default=config['n_concepts'],
                        help=f'Number of concepts to include (default '
                             f'{"all" if config["n_concepts"] == -1 else config["n_concepts"]})')
    parser.add_argument('--reduce-to-binary', type=str2bool, default=config['reduce_to_binary'],
                        help=f'Reduce dataset to binary (default {config["reduce_to_binary"]})')
    parser.add_argument('--basal-cell', type=str2bool, default=config['basal_cell'],
                        help=f'Reduce dataset to basal cell carcinoma (default {config["basal_cell"]})')
    parser.add_argument('--melanoma', type=str2bool, default=config['melanoma'],
                        help=f'Reduce dataset to melanoma (default {config["melanoma"]})')
    parser.add_argument('--balance-y-classes', type=str2bool, default=config['balance_y_classes'],
                        help=f'Balance y classes (default {config["balance_y_classes"]})')
    parser.add_argument('--drop-basal-cell', type=str2bool, default=config['drop_basal_cell'],
                        help=f'Drop basal cell carcinoma (default {config["drop_basal_cell"]})')
    parser.add_argument('--combine-train-val', type=str2bool, default=config['combine_train_val'],
                        help=f'Combine train and val sets (default {config["combine_train_val"]})')
    parser.add_argument('--majority-denoising', type=str2bool, default=config['majority_denoising'],
                        help=f'Denoise training set with majority voting (default {config["majority_denoising"]})')
    parser.add_argument('--augmentation-factor', type=float, default=config['augmentation_factor'],
                        help=f'Augmentation factor (default {config["augmentation_factor"]})')
    parser.add_argument('--suffix', type=str, default=config['suffix'],
                        help=f'Suffix to add to file name (default {config["suffix"]})')

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    meta_file = DERM7_META
    assert os.path.exists(meta_file), f"meta_file processed dir {meta_file} does not exist"
    train_idx_file = DERM7_TRAIN_IDX
    assert os.path.exists(train_idx_file), f"DERM7 train idx file {train_idx_file} does not exist"
    val_idx_file = DERM7_VAL_IDX
    assert os.path.exists(val_idx_file), f"DERM7 val idx file {val_idx_file} does not exist"

    save_dir = os.path.join(PROCESSED_DATASETS_DIR, 'derm7pt')
    if args.n_concepts > 0:
        save_dir = os.path.join(save_dir, f'{args.n_concepts}_concepts')
    if args.reduce_to_binary:
        save_dir = os.path.join(save_dir, 'binary')
    elif args.basal_cell:
        save_dir = os.path.join(save_dir, 'basal_cell')
    elif args.melanoma:
        save_dir = os.path.join(save_dir, 'melanoma')
    if args.drop_basal_cell:
        save_dir += '_no_basal_cell'
    if args.balance_y_classes:
        save_dir += '_balanced_y_classes'
    if args.majority_denoising:
        save_dir += '_majority_denoising'
    if args.combine_train_val:
        save_dir += '_combined_train_val'
    if args.augmentation_factor > 1.0:
        save_dir += f'_augmented_{args.augmentation_factor}'
    if args.suffix:
        save_dir += f'_{args.suffix}'

    train_set, val_set, test_set = process_derm7pt_datasets(meta_file, train_idx_file, val_idx_file,
                                                            n_concepts=args.n_concepts,
                                                            reduce_to_binary=args.reduce_to_binary,
                                                            basal_cell=args.basal_cell,
                                                            melanoma=args.melanoma,
                                                            balance_y_classes=args.balance_y_classes,
                                                            drop_basal_cell=args.drop_basal_cell,
                                                            majority_denoising=args.majority_denoising,
                                                            combine_train_val=args.combine_train_val,
                                                            augmentation_factor=args.augmentation_factor)


    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'training.pkl'), 'wb') as set_file:
        dill.dump(train_set, set_file, protocol=dill.HIGHEST_PROTOCOL)
    print('Saved dataset to {}'.format(os.path.join(save_dir, 'training.pkl')))

    with open(os.path.join(save_dir, 'validation.pkl'), 'wb') as set_file:
        dill.dump(val_set, set_file, protocol=dill.HIGHEST_PROTOCOL)
    print('Saved dataset to {}'.format(os.path.join(save_dir, 'validation.pkl')))

    with open(os.path.join(save_dir, 'test.pkl'), 'wb') as set_file:
        dill.dump(test_set, set_file, protocol=dill.HIGHEST_PROTOCOL)
    print('Saved dataset to {}'.format(os.path.join(save_dir, 'test.pkl')))
