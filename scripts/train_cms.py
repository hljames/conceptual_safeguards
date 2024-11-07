"""
Train ConceptModels for CUB
"""
import os
import argparse
import time

from constants import PROCESSED_DATASETS_DIR, SAVED_MODELS_DIR
from models.concept_model import train_concept_models
from utils import str2bool


def config():
    return {
        'dataset': 'cub/cub_v2_augmented_10.0',
        'embedding_backbone_name': 'inception',
        # General
        'max_samples': None,
        'random_state': 1,
        'batch_size': 16,
        'num_workers': os.cpu_count()-1,
        'model_type': 'mlp',
        'calibration_method': 'sigmoid',
        'persistent_workers': "True",
        'device': 'mps',
        'retrain_models': "True",
        'print_stats': "True",
        'suffix': 'calv2',
        'use_cached_predictions': "True",
        'balance_by_concept': "False",

        # LogReg
        'calibrate_logreg': "True",
        'logreg_solver': 'lbfgs',
        'logreg_penalty': 'none',
        'max_iters': 2000,
        # SVM
        'calibrate_svm': "False",
        'SVC_C': 1.0,

        # MLP
        'lr': 1e-3,
        'max_epochs': 100,
        'min_delta': 0.0,
        'patience': 3,
        'l1': 50,
        'calibrate_mlp': "True",

    }


if __name__ == '__main__':
    config = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=config['dataset'], help='Dataset to train on')
    parser.add_argument('--model-type', type=str, default=config['model_type'], help='Type of model to train')
    parser.add_argument('--calibrate-svm', type=str2bool, default=config['calibrate_svm'],
                        help='Whether to train calibrate SVM models')
    parser.add_argument('--calibrate-logreg', type=str2bool, default=config['calibrate_logreg'],
                        help='Whether to train calibrate logreg models')
    parser.add_argument('--calibrate-mlp', type=str2bool, default=config['calibrate_mlp'],
                        help='Whether to train calibrate MLP models')
    parser.add_argument('--calibration-method', type=str, default=config['calibration_method'],
                        help='Calibration method')
    parser.add_argument('--embedding-backbone-name', type=str, default=config['embedding_backbone_name'],
                        help='Backbone for embedding model')
    parser.add_argument('--SVC-C', type=float, default=config['SVC_C'], help='SVM C parameter')
    parser.add_argument('--logreg-solver', type=str, default=config['logreg_solver'], help='LogReg solver')
    parser.add_argument('--max-iters', type=int, default=config['max_iters'],
                        help='Maximum number of iterations for logreg')
    parser.add_argument('--logreg-penalty', type=str, default=config['logreg_penalty'], help='LogReg penalty')
    parser.add_argument('--use-cached-predictions', type=str2bool, default=config['use_cached_predictions'],
                        help='Whether to use cached predictions')
    parser.add_argument('--balance-by-concept', type=str2bool, default=config['balance_by_concept'],
                        help='Whether to balance by concept')
    parser.add_argument('--max-samples', default=config['max_samples'],
                        help='Maximum number of samples per concept label')
    parser.add_argument('--random-state', type=int, default=config['random_state'], help='Random state')
    parser.add_argument('--batch-size', type=int, default=config['batch_size'], help='Training batch size')
    parser.add_argument('--device', type=str, default=config['device'], help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=config['num_workers'],
                        help='Number of workers for data loaders')
    parser.add_argument('--persistent-workers', type=str2bool, default=config['persistent_workers'],
                        help='Whether to use persistent workers for data loaders')
    parser.add_argument('--lr', type=float, default=config['lr'], help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=config['max_epochs'],
                        help='Maximum number of epochs to train for')
    parser.add_argument('--min-delta', type=float, default=config['min_delta'], help='Minimum delta for early stopping')
    parser.add_argument('--patience', type=int, default=config['patience'], help='Patience for early stopping')
    parser.add_argument('--l1', type=float, default=config['l1'], help='Layer 1 size for concept models')
    parser.add_argument('--retrain-models', type=str2bool, default=config['retrain_models'],
                        help='Whether to retrain models')
    parser.add_argument('--print-stats', type=str2bool, default=config['print_stats'],
                        help='Whether to print stats')
    parser.add_argument('--suffix', type=str, default=config['suffix'],
                        help='Suffix for model save directory')

    args = parser.parse_args()
    data_dir = os.path.join(PROCESSED_DATASETS_DIR, args.dataset)
    assert os.path.exists(data_dir), data_dir
    # if 'synth' in data_dir:
    #     dataset_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    # else:
    #     dataset_dirs = [data_dir]

    dataset_dirs = [data_dir]
    model_save_dir = data_dir.replace(PROCESSED_DATASETS_DIR, SAVED_MODELS_DIR)
    os.makedirs(model_save_dir, exist_ok=True)
    # start training time
    start_time = time.time()
    assert len(dataset_dirs) > 0, dataset_dirs
    for dataset_dir in dataset_dirs:
        train_file_path = os.path.join(dataset_dir, 'training.pkl')
        validation_file_path = os.path.join(dataset_dir, 'validation.pkl')
        test_file_path = os.path.join(dataset_dir, 'test.pkl')
        concept_models_dir = os.path.join(model_save_dir, 'concept_models', args.embedding_backbone_name,
                                          args.model_type, )
        if args.balance_by_concept:
            concept_models_dir = concept_models_dir + '_balanced'
        if args.max_samples:
            concept_models_dir = concept_models_dir + '_ms{}'.format(args.max_samples)
        if args.SVC_C != 1.0 and args.model_type == 'SVM':
            concept_models_dir = concept_models_dir + '_SVC_C{}'.format(args.SVC_C)
        if args.model_type == 'logreg' and args.logreg_solver != 'lbfgs':
            concept_models_dir = concept_models_dir + f'_{args.logreg_solver}'
        if args.model_type == 'logreg' and args.logreg_penalty.lower() != 'l2':
            concept_models_dir = concept_models_dir + f'_p{args.logreg_penalty}'
        if (args.model_type == 'logreg' and args.calibrate_logreg) or (
                args.model_type == 'mlp' and args.calibrate_mlp) or (
                args.model_type == 'linearsvm' and args.calibrate_linearsvm):
            concept_models_dir = concept_models_dir + '_cal'
            # if args.calibration_method != 'isotonic':
            concept_models_dir = concept_models_dir + f'_{args.calibration_method}'
        concept_models_dir = concept_models_dir.replace('isotonic', 'iso')
        concept_models_dir = concept_models_dir.replace('sigmoid', 'sig')
        if args.suffix:
            concept_models_dir = concept_models_dir + '_' + args.suffix

        print('saving models to {}'.format(concept_models_dir))
        train_concept_models(train_file_path, validation_file_path,
                             test_file_path=test_file_path, concept_models_dir=concept_models_dir,
                             **vars(args))
        end_time = time.time()
        print(f"Time taken for processing: {end_time - start_time:.2f} seconds")
