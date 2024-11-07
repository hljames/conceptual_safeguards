"""
Generate datasets by as specified by the parameters concept_noise_probabilities, parity_inds, coefficients, intercept
"""
import argparse
import itertools
import os
from typing import List
import multiprocessing
import numpy as np
import dill
from datetime import datetime
import shutil
import pathlib

from numpy import ndarray
from tqdm import tqdm

from data.synthetic_data import *


def generate_all_datasets(data_base_dir: str,
                          n: int,
                          xp: List[float],
                          overwrite=False,
                          num_workers=1,
                          **kwargs) -> List[List[CBMDataset]]:
    """
    generate datsets using all combinations of the parameters in kwargs
    :return: datasets
    """
    assert os.path.exists(data_base_dir), data_base_dir
    date_time = datetime.now().strftime("synth_data_%m_%d_%H_%M")
    data_dir = os.path.join(data_base_dir, date_time)
    if os.path.exists(data_dir):
        assert overwrite, 'data_dir already exists, use --overwrite to overwrite'
        shutil.rmtree(data_dir)
    os.mkdir(data_dir)
    X = np.array([[float(bernoulli(p)) for p in xp] for _ in range(n)])
    params_dict = dict(kwargs)
    datasets = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        tasks = []
        for params in itertools.product(*params_dict.values()):
            task = pool.apply_async(create_dataset, (X, data_dir), dict(zip(params_dict.keys(), params)))
            tasks.append(task)
        for task in tqdm(tasks):
            datasets.append(task.get())
    print(f'{len(datasets)} datasets created, saved in {data_dir}')
    return datasets


def create_dataset(
        X: ndarray,
        data_dir: str,
        concept_noise_probs: List[float],
        # concept_dependence_matrix: List[List[int]],
        concept_dependence_prob: float,
        parity_inds: List[List[int]],
        coefs: List[float],
        intercept: float,
        regenerate_dataset=False) -> List[CBMDataset]:
    n_concepts = len(concept_noise_probs)
    assert n_concepts == len(parity_inds) == len(coefs)
    parity_str = "-".join([",".join([str(v) for v in par]) for par in parity_inds])
    concept_noise_prob_str = ",".join([str(cp) for cp in concept_noise_probs])
    coefs_str = ",".join([str(c) for c in coefs])
    dataset_desc = f'p{concept_noise_prob_str}_p{parity_str}_c{coefs_str}_i{intercept}_p_depen{concept_dependence_prob}'
    dataset_dir = os.path.join(data_dir, dataset_desc)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)
    datasets = ['training', 'validation', 'test']
    if regenerate_dataset or not os.path.exists(os.path.join(dataset_dir, 'training.pkl')):
        # create string description
        s = f'n samples: {len(X)}\n'
        for i in range(len(concept_noise_probs)):
            s += f'c{i + 1} = parity({", ".join([f"x{j}" for j in parity_inds[i]])}) w/ p={concept_noise_probs[i]}\n'
        s += f'p = logistic({" + ".join([f"{coefs[i]}c{i}" for i in range(len(concept_noise_probs))])} + ' \
             f'({intercept}))\n'
        s += f'y ~ Bernoulli(p)'

        # split X into train, validation, and test sets
        n_train = int(.6 * len(X))
        n_valid = int(.2 * len(X))
        indices = np.random.permutation(len(X))

        # split the data into train, validation, and test sets
        split_sizes = [n_train, n_valid, len(X) - n_train - n_valid]
        for i, dataset in enumerate(datasets):
            start_idx = sum(split_sizes[:i])
            end_idx = start_idx + split_sizes[i]
            idx = indices[start_idx:end_idx]
            X_set = X[idx]
            C_probas = np.array([[compute_c_i_proba(x, cp_noise, p_inds) for x in X_set] for
                        cp_noise, p_inds in
                        zip(concept_noise_probs, parity_inds)]).T
            C_set = np.array([[bernoulli(cp) for cp in c_proba] for c_proba in C_probas])
            y_proba_set = compute_prob_ys_given_C(C_set, coefs, intercept)
            y_set = np.array([bernoulli(yp) for yp in y_proba_set])
            set_obj = SyntheticCBMDataset(X=X_set,
                                 C=C_set,
                                 y=y_set,
                                 y_proba=y_proba_set,
                                 concept_noise_probs=concept_noise_probs,
                                 parity_inds=parity_inds,
                                 coefs=coefs,
                                 intercept=intercept,
                                 str_desc=s,
                                 concept_dependence=concept_dependence_prob)
            with open(os.path.join(dataset_dir, f"{dataset}.pkl"), 'wb') as set_file:
                dill.dump(set_obj, set_file, protocol=dill.HIGHEST_PROTOCOL)
    sets = []
    for dataset in datasets:
        with open(os.path.join(dataset_dir, f"{dataset}.pkl"), 'rb') as set_file:
            sets.append(dill.load(set_file))
    return sets


CONCEPT_NOISE_PROB_DEFAULT_VALS = np.arange(0., 1., 0.05)
CONCEPT_NOISE_PROB_DEFAULT = [[round(v, 2)] * 3 for v in CONCEPT_NOISE_PROB_DEFAULT_VALS]

CONCEPT_DEPENDENCE_DEFAULTS = [[[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]],  # no dependence between concepts

                               [[0, 0, 0],
                                [0, 0, 0],
                                [1, 0, 0]],  # dependence c2 -> c0

                               [[0, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0]],  # dependence c2 -> c0, c1 -> c0
                               ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', type=pathlib.Path, default='/Users/hailey/repos/uacbm/datasets',
                        help='base directory for data')

    parser.add_argument('--n_samples', type=int, default=100000, help='number of samples to generate')
    parser.add_argument('--x_probs', nargs='+', type=float, default=[.7] * 5, help='probabilities of x values')
    parser.add_argument('--overwrite', action='store_true', help='force the creation of new datasets')

    parser.add_argument('--concept-noise-probabilities', nargs='+', type=float,
                        default=CONCEPT_NOISE_PROB_DEFAULT,
                        help='probabilities of noise for each of the concepts')
    parser.add_argument('--concept-dependence-matrices', nargs='+', type=int,
                        default=CONCEPT_DEPENDENCE_DEFAULTS,
                        help='types of dependence between each concept')
    parser.add_argument('--concept-dependence-prob', nargs='+', type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help='types of dependence between each concept')
    parser.add_argument('--parity_inds', nargs='+', type=int, default=[[[0, 1, 3], [0, 1, 2], [0, 1, 4]]],
                        help='parity indices')
    parser.add_argument('--coefficients', nargs='+', type=float, default=[[1., 2., 3.]], help='coefficients')
    parser.add_argument('--intercept', nargs='+', type=float, default=[-2.], help='intercept')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 1, help='number of workers')

    args = parser.parse_args()
    concept_noise_probabilities = args.concept_noise_probabilities
    if isinstance(concept_noise_probabilities[0], float):
        concept_noise_probabilities = [[cp] * len(args.coefficients[0]) for cp in concept_noise_probabilities]

    generate_all_datasets(data_base_dir=args.data_base_dir,
                          n=args.n_samples,
                          xp=args.x_probs,
                          overwrite=args.overwrite,
                          concept_noise_probs=concept_noise_probabilities,
                          concept_dependence_prob=args.concept_dependence_prob,
                          # concept_dependence_matrix=args.concept_dependence_matrices,
                          parity_inds=args.parity_inds,
                          coefs=args.coefficients,
                          intercept=args.intercept,
                          num_workers=args.num_workers)
