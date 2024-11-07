"""
Train models on concept datasets
"""
import argparse
import dill
import numpy as np
import os
import time

import torch.multiprocessing

from constants import PROCESSED_DATASETS_DIR, SAVED_MODELS_DIR
from models.concept_bottleneck_model import train_concept_bottleneck_models
from models.concept_model import load_concept_models

from utils import str2bool

torch.multiprocessing.set_sharing_strategy('file_system')


def config():
    return {

        'dataset': 'cub/cub_augmented_10.0_superclasses',
        'concept_models': 'cub/cub_v2_augmented_10.0/concept_models/inception/logreg_pnone_cal_sig',
        'cbm_model_types': ['indepen', 'sequential', 'propagation'],  # 'cal_indepen'], # ece increases with cal_indepen
        'use_fast_propagation': "True",
        'concept_subset_desc': 'all',
        'use_cached_concept_probas': "True",
        'calibrate_y_model': "True",
        'calibration_method': 'sigmoid',
        'cal_on_preds': 'True',
        'lr_solver': 'lbfgs',
        'lr_penalty': None,
        # 'device': 'cpu',4
        'num_workers': os.cpu_count()-1, #1,
        'retrain': "True",
        'print_stats': "True",
        'include_test': "True",
        'suffix': '',
        'random_state': 1,
    }


CONCEPT_SUBSETS = {
    'all': 'all',
    'top5_f1': ["TypicalPigmentNetwork", "AbsentPigmentNetwork", "AbsentVascularStructures", "AbsentStreaks",
                "IrregularDG"],
    'top5_auc': ["CombinationRegressionStructures", "RegularStreaks", "RegularVascularStructures", "AbsentStreaks",
                 "AbsentDG"],
}

if __name__ == '__main__':
    config = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config['dataset'], help='Dataset for training CBMs')
    parser.add_argument('--concept-models', type=str, default=config['concept_models'],
                        help='Concept models for training CBMs')
    parser.add_argument('--use-fast-propagation', type=str2bool, default=config['use_fast_propagation'],
                        help='Whether to use fast propagation')
    # parser.add_argument('--calibrated-concept-models', type=str, default=config['calibrated_concept_models'],
    #                     help='Calibrated concept models for training CBMs')
    parser.add_argument('--cbm-model-types', type=str, nargs='+', default=config['cbm_model_types'],
                        help='Types of CBM models to train')
    parser.add_argument('--concept-subset-desc', type=str, default=config['concept_subset_desc'],
                        help='Subset of concepts to use for training CBMs (str desc)')
    parser.add_argument('--use-cached-concept-probas', type=str2bool, default=config['use_cached_concept_probas'],
                        help='Whether to use cached concept probabilities')
    parser.add_argument('--calibrate-y-model', type=str2bool, default=config['calibrate_y_model'],
                        help='Whether to calibrate y model')
    parser.add_argument('--calibration-method', type=str, default=config['calibration_method'],
                        help='Calibration method for y model')
    parser.add_argument('--cal-on-preds', type=str2bool, default=config['cal_on_preds'],
                        help='Whether to calibrate on predictions from model or calibration method')
    parser.add_argument('--random-state', type=int, default=config['random_state'],
                        help='Random state for training CBMs')
    parser.add_argument('--lr-solver', type=str, default=config['lr_solver'],
                        help='Solver for logistic regression')
    parser.add_argument('--lr-penalty', type=str, default=config['lr_penalty'],
                        help='Penalty for logistic regression')
    parser.add_argument('--retrain', type=str2bool, default=config['retrain'],
                        help='Whether to retrain models')
    parser.add_argument('--print-stats', type=str2bool, default=config['print_stats'],
                        help='Whether to print stats')
    parser.add_argument('--include-test', type=str2bool, default=config['include_test'],
                        help='Whether to include test set')
    parser.add_argument('--suffix', type=str, default=config['suffix'],
                        help='Suffix for model save directory')
    parser.add_argument('--num-workers', type=int, default=config['num_workers'],
                        help='Number of workers for training CBMs')

    args = parser.parse_args()

    # construct directories and check existence
    all_datasets_dir = os.path.join(PROCESSED_DATASETS_DIR, args.dataset)
    assert os.path.exists(all_datasets_dir), all_datasets_dir

    # get dataset directories
    # if 'synth' in args.dataset:
    #     dataset_dirs = [f.path for f in os.scandir(all_datasets_dir) if f.is_dir()]
    # else:
    #     dataset_dirs = [all_datasets_dir]
    dataset_dirs = [all_datasets_dir]
    assert len(dataset_dirs) > 0, dataset_dirs
    start_time = time.time()
    for dataset_dir in dataset_dirs:
        assert os.path.exists(dataset_dir), dataset_dir

        # check existence of concept models
        concept_models_dir = os.path.join(SAVED_MODELS_DIR, args.concept_models)
        assert os.path.exists(concept_models_dir), concept_models_dir

        # construct cbm save directory
        cbms_save_dir = os.path.join(dataset_dir.replace(PROCESSED_DATASETS_DIR, SAVED_MODELS_DIR),
                                     'cbms')

        print(f'training models for dataset {dataset_dir}')
        train_file_path = os.path.join(dataset_dir, 'training.pkl')
        validation_file_path = os.path.join(dataset_dir, 'validation.pkl')
        if args.include_test:
            test_file_path = os.path.join(dataset_dir, 'test.pkl')
        else:
            test_file_path = None
        with open(train_file_path, 'rb') as train_file:
            training_dataset = dill.load(train_file)
            multiclass = len(np.unique(training_dataset.y)) > 2
        cbms_save_dir += f'_cms_{os.path.basename(args.concept_models)}'
        cms = load_concept_models(concept_models_dir)
        backbone_name = cms[0].embedding_backbone_name
        if backbone_name is not None:
            cbms_save_dir += f'_{backbone_name}'
        if args.calibrate_y_model:
            assert args.calibration_method in ['sigmoid', 'isotonic']
            cbms_save_dir = cbms_save_dir + '_ycal'
            if args.cal_on_preds:
                cbms_save_dir = cbms_save_dir + '_onpreds'
            cbms_save_dir = cbms_save_dir + '_' + args.calibration_method
        cbms_save_dir = cbms_save_dir.replace('sigmoid', 'sig')
        cbms_save_dir = cbms_save_dir.replace('isotonic', 'iso')
        if args.concept_subset_desc != 'all':
            cbms_save_dir = cbms_save_dir + '_' + args.concept_subset_desc
        if args.lr_solver != 'lbfgs':
            cbms_save_dir = cbms_save_dir + '_' + args.lr_solver
        if args.lr_penalty is not None and args.lr_penalty != 'none':
            cbms_save_dir = cbms_save_dir + '_' + args.lr_penalty
        print('saving models to {}'.format(cbms_save_dir))
        train_concept_bottleneck_models(train_file_path,
                                        validation_file_path,
                                        args.cbm_model_types,
                                        cbms_save_dir,
                                        concept_models_dir,
                                        # calibrated_concept_models_dir,
                                        args.retrain,
                                        args.print_stats,
                                        concept_subset=CONCEPT_SUBSETS[args.concept_subset_desc],
                                        use_fast_propagation=args.use_fast_propagation,
                                        test_file_path=test_file_path,
                                        multiclass=multiclass,
                                        random_state=args.random_state,
                                        lr_solver=args.lr_solver,
                                        lr_penalty=args.lr_penalty,
                                        use_cached_concept_probas=args.use_cached_concept_probas,
                                        calibrate_y_model=args.calibrate_y_model,
                                        calibration_method=args.calibration_method,
                                        cal_on_preds=args.cal_on_preds,
                                        num_workers=args.num_workers,
                                        )
