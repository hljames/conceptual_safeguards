"""
Modify processed CUB datasets (e.g., binarize)
"""

import os
from copy import deepcopy

import dill

from collections import Counter

import numpy as np
from prettytable import PrettyTable
from sklearn.linear_model import LogisticRegression

from constants import PROCESSED_DATASETS_DIR, CUB_DATA_DIR
from data.image_data import augment_training_dataset
from metrics import calculate_metrics

orig_cub_dir = os.path.join(PROCESSED_DATASETS_DIR, 'cub', 'cub_v2_augmented_10.0')
print_raw_dataset_prediction_info = False

dset_type = 'superclasses'
dset_types = ['binarize_datasets_c0_c1', 'binarize_datasets_warbler_sparrow',
              'binarize_datasets_warbler_sparrow_vs_rest', 'binarize_dataset_top10', 'binarize_dataset_bottom60',
              'superclasses']
assert dset_type in dset_types

augment_training_data = False
AUG_FACTOR = 1.0

training_set = dill.load(open(os.path.join(orig_cub_dir, 'training.pkl'), 'rb'))
validation_set = dill.load(open(os.path.join(orig_cub_dir, 'validation.pkl'), 'rb'))
test_set = dill.load(open(os.path.join(orig_cub_dir, 'test.pkl'), 'rb'))

if augment_training_data:
    orig_img_base_dir = os.path.join(CUB_DATA_DIR, 'images')
    aug_save_dir = os.path.join(CUB_DATA_DIR, 'augmented_images')
    print(f'Augmenting training dataset, size before augmentation: {len(training_set)}')
    train_dataset_augmented = augment_training_dataset(training_set,
                                                       augmentation_factor=AUG_FACTOR)
    print(f'Augmented training dataset size: {len(train_dataset_augmented)}')

    # save new datasets
    augmented_data_dir = orig_cub_dir + f'_augmented_{AUG_FACTOR}'
    os.makedirs(augmented_data_dir, exist_ok=True)
    dill.dump(train_dataset_augmented, open(os.path.join(augmented_data_dir, 'training.pkl'), 'wb'))
    dill.dump(validation_set, open(os.path.join(augmented_data_dir, 'validation.pkl'), 'wb'))
    dill.dump(test_set, open(os.path.join(augmented_data_dir, 'test.pkl'), 'wb'))
    print(f'Saved augmented datasets to {augmented_data_dir}')

classes_desc_file = os.path.join(CUB_DATA_DIR, 'classes.txt')
# read in the classes file:
nums_classes_dict = {}
with open(classes_desc_file, 'r') as f:
    for line in f:
        class_num, class_name_desc = line.strip().split(' ')
        class_name = class_name_desc.split('.')[1]
        general_name = class_name.split('_')[-1]
        nums_classes_dict[int(class_num)] = (class_name, general_name)

general_names_counts = Counter([v[1] for v in nums_classes_dict.values()])
warbler_class_nums = [k for k, v in nums_classes_dict.items() if v[1] == 'Warbler']
sparrow_class_nums = [k for k, v in nums_classes_dict.items() if v[1] == 'Sparrow']

# Note: All classes are roughly the same size (29-30 samples)  and we are able
# to perfectly predict the outcome with ground truth concepts for each class
if print_raw_dataset_prediction_info:
    pt = PrettyTable()
    datasets = {
        'train': training_set,
        'validation': validation_set,
        'test': test_set
    }
    pt.field_names = ["Dataset", "Class", "n_samples", "log_loss", "error", "ece", "f1", "auc_roc"]

    model = LogisticRegression(penalty=None, solver='lbfgs', multi_class='multinomial')
    model.fit(training_set.C, training_set.y)
    for dataset_name, dataset in datasets.items():
        y_true = dataset.y
        y_pred = model.predict(dataset.C)
        y_pred_proba = model.predict_proba(dataset.C)

        unique_labels = np.unique(y_true)

        for label in unique_labels:
            y_true_label = (y_true == label).astype(int)
            y_pred_label = (y_pred == label).astype(int)

            metrics = calculate_metrics(y_true_label, y_pred_label, y_pred_proba[:, label], labels=np.array([0, 1]))

            pt.add_row([dataset_name, label, metrics['n_samples'],
                        f"{metrics['log_loss']:.4f}", f"{metrics['error']:.4f}",
                        metrics['ece'], f"{metrics['f1']:.4f}", f"{metrics['auc_roc']:.4f}"])
    print(pt)

    # print a table of metrics for each class for training, validation, and test sets

    all_ys = np.concatenate([training_set.y, validation_set.y, test_set.y])
    class_counts = Counter(all_ys)

    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_label, count in sorted_class_counts:
        print(f"Class {class_label}: {count} instances")

if dset_type == 'binarize_datasets_c0_c1':
    # binarize the datasets by making y = 1 for the first class and y = 0 for all others
    training_set_c0 = deepcopy(training_set)
    validation_set_c0 = deepcopy(validation_set)
    test_set_c0 = deepcopy(test_set)
    training_set_c0.y = np.where(training_set_c0.y == 0, 1, 0)
    validation_set_c0.y = np.where(validation_set_c0.y == 0, 1, 0)
    test_set_c0.y = np.where(test_set_c0.y == 0, 1, 0)

    # save the binarized datasets
    binarized_cub_dir_c0 = orig_cub_dir + f'_binary_c0'
    os.makedirs(binarized_cub_dir_c0, exist_ok=True)
    dill.dump(training_set_c0, open(os.path.join(binarized_cub_dir_c0, 'training.pkl'), 'wb'))
    dill.dump(validation_set_c0, open(os.path.join(binarized_cub_dir_c0, 'validation.pkl'), 'wb'))
    dill.dump(test_set_c0, open(os.path.join(binarized_cub_dir_c0, 'test.pkl'), 'wb'))

    # binarize the datasets by making y = 1 for the second class and y = 0 for all others
    training_set_c1 = deepcopy(training_set)
    validation_set_c1 = deepcopy(validation_set)
    test_set_c1 = deepcopy(test_set)
    #
    training_set_c1.y = np.where(training_set_c1.y == 1, 1, 0)
    validation_set_c1.y = np.where(validation_set_c1.y == 1, 1, 0)
    test_set_c1.y = np.where(test_set_c1.y == 1, 1, 0)

    # save the binarized datasets
    binarized_cub_dir_c1 = orig_cub_dir + f'_binary_c1'
    os.makedirs(binarized_cub_dir_c1, exist_ok=True)
    dill.dump(training_set_c1, open(os.path.join(binarized_cub_dir_c1, 'training.pkl'), 'wb'))
    dill.dump(validation_set_c1, open(os.path.join(binarized_cub_dir_c1, 'validation.pkl'), 'wb'))
    dill.dump(test_set_c1, open(os.path.join(binarized_cub_dir_c1, 'test.pkl'), 'wb'))

elif dset_type == 'binarize_datasets_warbler_sparrow':
    # binarize the datasets by making y = 1 for the warblers and y = 0 for all others
    training_set_warbler = deepcopy(training_set)
    validation_set_warbler = deepcopy(validation_set)
    test_set_warbler = deepcopy(test_set)
    training_set_warbler.y = np.where(np.isin(training_set_warbler.y, warbler_class_nums), 1, 0)
    validation_set_warbler.y = np.where(np.isin(validation_set_warbler.y, warbler_class_nums), 1, 0)
    test_set_warbler.y = np.where(np.isin(test_set_warbler.y, warbler_class_nums), 1, 0)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(training_set_warbler.C, training_set_warbler.y)
    train_metrics_dict = calculate_metrics(training_set_warbler.y,
                                           model.predict(training_set_warbler.C),
                                           model.predict_proba(training_set_warbler.C)[:, 1])
    val_metrics_dict = calculate_metrics(validation_set_warbler.y,
                                         model.predict(validation_set_warbler.C),
                                         model.predict_proba(validation_set_warbler.C)[:, 1])
    test_metrics_dict = calculate_metrics(test_set_warbler.y,
                                          model.predict(test_set_warbler.C),
                                          model.predict_proba(test_set_warbler.C)[:, 1])
    print('Training metrics:')
    print(train_metrics_dict)
    print('Validation metrics:')
    print(val_metrics_dict)
    print('Test metrics:')
    print(test_metrics_dict)

    # save the binarized datasets
    binarized_cub_dir_warbler = orig_cub_dir + f'_binary_warbler'
    os.makedirs(binarized_cub_dir_warbler, exist_ok=True)
    dill.dump(training_set_warbler, open(os.path.join(binarized_cub_dir_warbler, 'training.pkl'), 'wb'))
    dill.dump(validation_set_warbler, open(os.path.join(binarized_cub_dir_warbler, 'validation.pkl'), 'wb'))
    dill.dump(test_set_warbler, open(os.path.join(binarized_cub_dir_warbler, 'test.pkl'), 'wb'))
    print(f'Saved binarized datasets to {binarized_cub_dir_warbler}')

    # binarize the datasets by making y = 1 for sparrows and y = 0 for all others
    training_set_sparrow = deepcopy(training_set)
    validation_set_sparrow = deepcopy(validation_set)
    test_set_sparrow = deepcopy(test_set)
    training_set_sparrow.y = np.where(np.isin(training_set_sparrow.y, sparrow_class_nums), 1, 0)
    validation_set_sparrow.y = np.where(np.isin(validation_set_sparrow.y, sparrow_class_nums), 1, 0)
    test_set_sparrow.y = np.where(np.isin(test_set_sparrow.y, sparrow_class_nums), 1, 0)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(training_set_sparrow.C, training_set_sparrow.y)
    train_metrics_dict = calculate_metrics(training_set_sparrow.y,
                                           model.predict(training_set_sparrow.C),
                                           model.predict_proba(training_set_sparrow.C)[:, 1])
    val_metrics_dict = calculate_metrics(validation_set_sparrow.y,
                                         model.predict(validation_set_sparrow.C),
                                         model.predict_proba(validation_set_sparrow.C)[:, 1])
    test_metrics_dict = calculate_metrics(test_set_sparrow.y,
                                          model.predict(test_set_sparrow.C),
                                          model.predict_proba(test_set_sparrow.C)[:, 1])
    print('Training metrics:')
    print(train_metrics_dict)
    print('Validation metrics:')
    print(val_metrics_dict)
    print('Test metrics:')
    print(test_metrics_dict)

    # save the binarized datasets
    binarized_cub_dir_sparrow = orig_cub_dir + f'_binary_sparrow'
    os.makedirs(binarized_cub_dir_sparrow, exist_ok=True)
    dill.dump(training_set_sparrow, open(os.path.join(binarized_cub_dir_sparrow, 'training.pkl'), 'wb'))
    dill.dump(validation_set_sparrow, open(os.path.join(binarized_cub_dir_sparrow, 'validation.pkl'), 'wb'))
    dill.dump(test_set_sparrow, open(os.path.join(binarized_cub_dir_sparrow, 'test.pkl'), 'wb'))
    print(f'Saved binarized datasets to {binarized_cub_dir_sparrow}')

elif dset_type == 'binarize_datasets_warbler_sparrow_vs_rest':
    # binarize the datasets by making y = 1 for sparrows and warblers and y = 0 for all others
    training_set_warbler_sparrow = deepcopy(training_set)
    validation_set_warbler_sparrow = deepcopy(validation_set)
    test_set_warbler_sparrow = deepcopy(test_set)
    training_set_warbler_sparrow.y = np.where(np.isin(training_set_warbler_sparrow.y, warbler_class_nums + sparrow_class_nums), 1, 0)
    validation_set_warbler_sparrow.y = np.where(np.isin(validation_set_warbler_sparrow.y, warbler_class_nums + sparrow_class_nums), 1, 0)
    test_set_warbler_sparrow.y = np.where(np.isin(test_set_warbler_sparrow.y, warbler_class_nums + sparrow_class_nums), 1, 0)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(training_set_warbler_sparrow.C, training_set_warbler_sparrow.y)
    train_metrics_dict = calculate_metrics(training_set_warbler_sparrow.y,
                                             model.predict(training_set_warbler_sparrow.C),
                                                model.predict_proba(training_set_warbler_sparrow.C)[:, 1])
    val_metrics_dict = calculate_metrics(validation_set_warbler_sparrow.y,
                                             model.predict(validation_set_warbler_sparrow.C),
                                                model.predict_proba(validation_set_warbler_sparrow.C)[:, 1])
    test_metrics_dict = calculate_metrics(test_set_warbler_sparrow.y,
                                                model.predict(test_set_warbler_sparrow.C),
                                                model.predict_proba(test_set_warbler_sparrow.C)[:, 1])
    print('Training metrics:')
    print(train_metrics_dict)
    print('Validation metrics:')
    print(val_metrics_dict)
    print('Test metrics:')
    print(test_metrics_dict)

elif dset_type == 'binarize_dataset_top10':
    # pos_general_names = ['Warbler', 'Sparrow', 'Gull', 'Flycatcher', 'Tern', 'Vireo', 'Wren', 'Woodpecker',
    #                      'Kingfisher', 'Auklet',]
    sorted_general_names_counts = sorted(general_names_counts.items(), key=lambda x: x[1])

    top_10_general_names = [k for k, v in sorted_general_names_counts[-10:]]

    pos_names = [k for k, v in nums_classes_dict.items() if v[1] in top_10_general_names]
    print(f'Fraction of samples in  {top_10_general_names}: {np.mean(np.isin(training_set.y, pos_names))}')
    training_set_top_10 = deepcopy(training_set)
    validation_set_top_10 = deepcopy(validation_set)
    test_set_top_10 = deepcopy(test_set)
    training_set_top_10.y = np.where(np.isin(training_set_top_10.y, pos_names), 1, 0)
    validation_set_top_10.y = np.where(np.isin(validation_set_top_10.y, pos_names), 1, 0)
    test_set_top_10.y = np.where(np.isin(test_set_top_10.y, pos_names), 1, 0)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(training_set_top_10.C, training_set_top_10.y)
    train_metrics_dict = calculate_metrics(training_set_top_10.y,
                                             model.predict(training_set_top_10.C),
                                                model.predict_proba(training_set_top_10.C)[:, 1])
    val_metrics_dict = calculate_metrics(validation_set_top_10.y,
                                                model.predict(validation_set_top_10.C),
                                                model.predict_proba(validation_set_top_10.C)[:, 1])
    test_metrics_dict = calculate_metrics(test_set_top_10.y,
                                                model.predict(test_set_top_10.C),
                                                model.predict_proba(test_set_top_10.C)[:, 1])
    print('Training metrics:')
    print(train_metrics_dict)
    print('Validation metrics:')
    print(val_metrics_dict)
    print('Test metrics:')
    print(test_metrics_dict)

    binarized_cub_dir_top10 = orig_cub_dir + '_binary_top10'
    os.makedirs(binarized_cub_dir_top10, exist_ok=True)
    dill.dump(training_set_top_10, open(os.path.join(binarized_cub_dir_top10, 'training.pkl'), 'wb'))
    dill.dump(validation_set_top_10, open(os.path.join(binarized_cub_dir_top10, 'validation.pkl'), 'wb'))
    dill.dump(test_set_top_10, open(os.path.join(binarized_cub_dir_top10, 'test.pkl'), 'wb'))
    print(f'Saved binarized datasets to {binarized_cub_dir_top10}')

elif  dset_type == 'binarize_dataset_bottom60':
    # sort general_names_counts by value into list of tuples
    sorted_general_names_counts = sorted(general_names_counts.items(), key=lambda x: x[1])
    # get top 10 (last 10)
    bottom_60_general_names = [k for k, v in sorted_general_names_counts[:60]]
    pos_names = [k for k, v in nums_classes_dict.items() if v[1] in bottom_60_general_names]
    print(f'Fraction of samples in  bottom_60_general_names: {np.mean(np.isin(training_set.y, pos_names))}')
    training_set_bottom_60 = deepcopy(training_set)
    validation_set_bottom_60 = deepcopy(validation_set)
    test_set_bottom_60 = deepcopy(test_set)
    training_set_bottom_60.y = np.where(np.isin(training_set_bottom_60.y, pos_names), 1, 0)
    validation_set_bottom_60.y = np.where(np.isin(validation_set_bottom_60.y, pos_names), 1, 0)
    test_set_bottom_60.y = np.where(np.isin(test_set_bottom_60.y, pos_names), 1, 0)
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(training_set_bottom_60.C, training_set_bottom_60.y)
    train_metrics_dict = calculate_metrics(training_set_bottom_60.y,
                                                model.predict(training_set_bottom_60.C),
                                                model.predict_proba(training_set_bottom_60.C)[:, 1])
    val_metrics_dict = calculate_metrics(validation_set_bottom_60.y,
                                                model.predict(validation_set_bottom_60.C),
                                                model.predict_proba(validation_set_bottom_60.C)[:, 1])
    test_metrics_dict = calculate_metrics(test_set_bottom_60.y,
                                                model.predict(test_set_bottom_60.C),
                                                model.predict_proba(test_set_bottom_60.C)[:, 1])
    print('Training metrics:')
    print(train_metrics_dict)
    print('Validation metrics:')
    print(val_metrics_dict)
    print('Test metrics:')
    print(test_metrics_dict)

    binarized_cub_dir_bottom60 = orig_cub_dir + '_binary_bottom60'
    os.makedirs(binarized_cub_dir_bottom60, exist_ok=True)
    dill.dump(training_set_bottom_60, open(os.path.join(binarized_cub_dir_bottom60, 'training.pkl'), 'wb'))
    dill.dump(validation_set_bottom_60, open(os.path.join(binarized_cub_dir_bottom60, 'validation.pkl'), 'wb'))
    dill.dump(test_set_bottom_60, open(os.path.join(binarized_cub_dir_bottom60, 'test.pkl'), 'wb'))
    print(f'Saved binarized datasets to {binarized_cub_dir_bottom60}')

elif dset_type == 'superclasses':
    superclass_names = sorted(list(set([v[1] for k, v in nums_classes_dict.items()])))
    superclass_to_ind = {k: i for i, k in enumerate(superclass_names)}
    superclass_all_inds = np.array(sorted(list(set(superclass_to_ind.values()))))
    class_to_superclass = {k: superclass_to_ind[v[1]] for i,(k, v) in enumerate(nums_classes_dict.items())}
    training_set_superclasses = deepcopy(training_set)
    validation_set_superclasses = deepcopy(validation_set)
    test_set_superclasses = deepcopy(test_set)
    training_set_superclasses.y = np.asarray([class_to_superclass[y+1] for y in training_set_superclasses.y])
    validation_set_superclasses.y = np.asarray([class_to_superclass[y+1] for y in validation_set_superclasses.y])
    test_set_superclasses.y = np.asarray([class_to_superclass[y+1] for y in test_set_superclasses.y])
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, multi_class='multinomial')
    model.fit(training_set_superclasses.C, training_set_superclasses.y)
    train_metrics_dict = calculate_metrics(training_set_superclasses.y,
                                                model.predict(training_set_superclasses.C),
                                                model.predict_proba(training_set_superclasses.C),
                                           labels=superclass_all_inds)
    val_metrics_dict = calculate_metrics(validation_set_superclasses.y,
                                                model.predict(validation_set_superclasses.C),
                                                model.predict_proba(validation_set_superclasses.C),
                                         labels=superclass_all_inds)
    test_metrics_dict = calculate_metrics(test_set_superclasses.y,
                                                model.predict(test_set_superclasses.C),
                                                model.predict_proba(test_set_superclasses.C),
                                          labels=superclass_all_inds)
    print('Training metrics:')
    print(train_metrics_dict)
    print('Validation metrics:')
    print(val_metrics_dict)
    print('Test metrics:')
    print(test_metrics_dict)

    cub_dir_superclasses = orig_cub_dir + '_superclasses'
    os.makedirs(cub_dir_superclasses, exist_ok=True)
    dill.dump(training_set_superclasses, open(os.path.join(cub_dir_superclasses, 'training.pkl'), 'wb'))
    dill.dump(validation_set_superclasses, open(os.path.join(cub_dir_superclasses, 'validation.pkl'), 'wb'))
    dill.dump(test_set_superclasses, open(os.path.join(cub_dir_superclasses, 'test.pkl'), 'wb'))
    print(f'Saved binarized datasets to {cub_dir_superclasses}')




