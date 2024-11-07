"""
Dataset Class definitions
"""
import numpy as np
import torch

from copy import deepcopy
from typing import List, Optional

from numpy import ndarray
from torch.utils.data import Dataset


class CBMDataset(Dataset):
    def __init__(self,
                 X: List,
                 C: ndarray,
                 y: ndarray,
                 **kwargs):
        assert len(X)
        assert y.ndim == 1, f'expected y.ndim 1, got y.ndim {y.ndim}'
        assert len(X) == y.shape[0], f'X.shape[0] {len(X)} != y.shape[0] {y.shape[0]}'
        assert len(X) == C.shape[0], f'X.shape[0] {len(X)} != C.shape[0] {C.shape[0]}'
        self.X = X
        self.C = C
        self.y = y
        self.is_binary = True if len(np.unique(self.y)) == 2 else False
        self._concept_names = kwargs.get('concept_names', None)
        self.__dict__.update(kwargs)
        self._str_desc = kwargs.get('str_desc', '')

    @property
    def str_desc(self):
        return self._str_desc if self._str_desc else f'n samples: {self.n}\n'

    @property
    def n(self):
        return len(self.X)

    @property
    def n_concepts(self):
        return self.C.shape[1]

    @property
    def concept_names(self):
        if hasattr(self, '_concept_names') and self._concept_names is not None:
            return self._concept_names
        else:
            return [f'c{i}' for i in range(self.n_concepts)]

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return self.str_desc

    def __desc__(self):
        return self.str_desc

    def __getitem__(self, idx):
        """
        Get item by index
        :param idx:
        :return:
        """
        self.X = np.asarray(self.X)
        X_idx = torch.from_numpy(np.array(self.X[idx], dtype=np.float32))
        C_idx = torch.from_numpy(np.array(self.C[idx, :], dtype=np.int32))
        y_idx = torch.from_numpy(np.array(self.y[idx], dtype=np.int32))
        return X_idx, C_idx, y_idx


def downsample_by_concept(dataset, concept_index: int):
    """
    Remove samples for which the concept is not know (e.g., -1)
    :param dataset:
    :param concept_index:
    :return:
    """
    dataset = deepcopy(dataset)
    # idx where dataset.C[:, concept_index] is not nan
    idx = np.where(~np.isnan(dataset.C[:, concept_index]))[0]
    dataset.X = [dataset.X[i] for i in idx]
    dataset.C = dataset.C[idx, :]
    dataset.y = dataset.y[idx]
    dataset._str_desc = f'n samples after downsampling: {len(dataset.X)}\n'
    return dataset


def downsample_by_max_samples(dataset, max_samples):
    """
    Downsample dataset to max_samples
    :param dataset:
    :param max_samples:
    :return:
    """
    dataset = deepcopy(dataset)
    idx = np.random.choice(np.arange(len(dataset.X)), max_samples, replace=False)
    dataset.X = [dataset.X[i] for i in idx]
    dataset.C = dataset.C[idx, :]
    dataset.y = dataset.y[idx]
    dataset._str_desc = f'n samples after downsampling: {len(dataset.X)}\n'
    return dataset


def downsample_by_index(dataset, idx):
    """
    Downsample dataset by index
    :param dataset:
    :param idx:
    :return:
    """
    dataset = deepcopy(dataset)
    dataset.X = [dataset.X[i] for i in idx]
    dataset.C = dataset.C[idx, :]
    dataset.y = dataset.y[idx]
    dataset._str_desc = f'n samples after downsampling: {len(dataset.X)}\n'
    return dataset


def downsample_to_balance_concept(dataset, concept_index: int, max_samples: Optional[int] = None):
    """
    Downsample dataset to include an equal number of samples for each concept class
    :param dataset:
    :param concept_index:
    :param max_samples:
    :return:
    """
    dataset = deepcopy(dataset)
    unique_concepts, counts = np.unique(dataset.C[:, concept_index], return_counts=True)

    # If max_samples is provided, use the minimum of max_samples/unique_concepts and the least frequent concept count
    # Otherwise, just use the least frequent concept count
    min_count = int(min(np.min(counts),
                        max_samples / len(unique_concepts)) if max_samples is not None else np.min(counts))

    new_X = []
    new_C = []
    new_y = []

    for uc in unique_concepts:
        idxs = np.where(dataset.C[:, concept_index] == uc)[0]

        # Limit the number of samples to min_count for each unique concept
        chosen_idxs = np.random.choice(idxs, min_count, replace=False)

        new_X.extend(dataset.X[chosen_idx] for chosen_idx in chosen_idxs)
        new_C.extend(dataset.C[chosen_idx] for chosen_idx in chosen_idxs)
        new_y.extend(dataset.y[chosen_idx] for chosen_idx in chosen_idxs)

    dataset.X = new_X
    dataset.C = np.array(new_C)
    dataset.y = np.array(new_y)
    dataset._str_desc = f'n samples after downsampling: {len(dataset.X)}\n'
    return dataset


def downsample_to_balance_y_classes(dataset, max_samples: Optional[int] = None):
    """
    Downsample dataset to include an equal number of samples for each y class
    :param dataset:
    :param max_samples:
    :return:
    """
    dataset = deepcopy(dataset)
    unique_y, counts = np.unique(dataset.y, return_counts=True)

    # If max_samples is provided, use the minimum of max_samples/unique_y and the least frequent y count
    # Otherwise, just use the least frequent y count
    min_count = int(min(np.min(counts),
                        max_samples / len(unique_y)) if max_samples is not None else np.min(counts))

    new_X = []
    new_C = []
    new_y = []

    for uc in unique_y:
        idxs = np.where(dataset.y == uc)[0]

        # Limit the number of samples to min_count for each unique concept
        chosen_idxs = np.random.choice(idxs, min_count, replace=False)

        new_X.extend(dataset.X[chosen_idx] for chosen_idx in chosen_idxs)
        new_C.extend(dataset.C[chosen_idx] for chosen_idx in chosen_idxs)
        new_y.extend(dataset.y[chosen_idx] for chosen_idx in chosen_idxs)

    dataset.X = new_X
    dataset.C = np.array(new_C)
    dataset.y = np.array(new_y)
    dataset._str_desc = f'n samples after downsampling: {len(dataset.X)}\n'
    return dataset


def majority_denoise_dataset(dataset):
    """
    Denoise the dataset by flipping concepts to the majority concept for each class
    Note: should be performed on training sets only
    :param dataset:
    :return:
    """
    dataset = deepcopy(dataset)
    unique_y = np.unique(dataset.y)

    majority_concepts = {}

    for y_class in unique_y:
        idxs = np.where(dataset.y == y_class)[0]
        C_subset = dataset.C[idxs, :]
        majority_concepts[y_class] = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=C_subset)

    for i in range(dataset.n):
        dataset.C[i, :] = majority_concepts[dataset.y[i]]

    dataset._str_desc = f'n samples after denoising: {dataset.n}\n'

    return dataset

