"""
Methods for constructing a synthetic dataset
"""
import math
import numpy as np
import torch

from numpy import ndarray
from typing import List, Optional
from numpy.testing import assert_almost_equal

from utils import generate_all_binary_vectors
from data.data import CBMDataset

class SyntheticCBMDataset(CBMDataset):
    def __init__(self,
                 X: List,
                 C: ndarray,
                 y: ndarray,
                 **kwargs):
        super().__init__(X, C, y, **kwargs)
        assert self.X.ndim == 2, f'expected X.ndim 2, got X.ndim {self.X.ndim}'
        self.X = self.X.astype(np.float32)  # torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = self.y.astype(np.float32)  # torch.from_numpy(np.array(y, dtype=np.int))
        self.n_features = self.X.shape[1]
        assert self.C.ndim == 2, f'expected C.ndim 2, got self.C.ndim {self.C.ndim}'
        assert self.C.shape[0] == self.n, f'self.C.shape[0] {self.C.shape[0]} != self.n {self.n}'
        assert self.C.shape[
                   1] == self.n_concepts, f'self.C.shape[1] {self.C.shape[1]} != self.n_concepts {self.n_concepts}'
        self.y_proba = np.array(kwargs.get('y_proba', None), dtype=np.float32)

    def downsample(self, max_samples: int):
        """
        Downsample the dataset to max_samples
        :param max_samples:
        :return:
        """
        if max_samples >= len(self.X):
            print(f'Warning: max_samples {max_samples} >= len(self.X) {len(self.X)}, not downsampling')
            return
        indices = np.random.choice(len(self.X), max_samples, replace=False)

        self.X = self.X[indices, :]
        self.y = self.y[indices]
        self.C = self.C[indices, :]

        if self.y_proba is not None:
            self.y_proba = self.y_proba[indices]


def bernoulli(p: float) -> bool:
    """
    Perform Bernoulli trial with probability p
    :param p: probabilitiy
    :return: binary outcome
    """
    return np.random.random() < p


def parity(lst: List[int]) -> int:
    """
    True if the sum of the bits is odd
    :param lst: list of binary bits
    :return: binary outcome
    """
    return int(sum(lst) % 2)


def logistic(x: float) -> float:
    """
    sigmoid/logistic function
    :param x: sigmoid parameter
    :return:
    """
    return 1 / (1 + math.exp(-x))


def compute_c_i_proba(x: ndarray, p_noise: float, p_inds: List[int]) -> float:
    """
    Compute the probability of a concept being true
    :param x: features
    :param p_noise: 1 - probability of using parity function for concepts
    :param p_inds: indices from features for parity functions
    :return:
    """
    # equal likely to be 0 or 1 with probability concept_noise_probs
    cp = (1 * 0.5 + 0 * 0.5) * p_noise
    # parity with probability 1 - concept_noise_probs
    cp += parity(x[p_inds]) * (1 - p_noise)
    return cp


def update_c_i_probas(prob_cs: List[float], dependence: [List[List[int]]]):
    prob_cs = prob_cs.copy()
    dependence = np.array(dependence)
    assert len(dependence) == len(prob_cs), "dependence must be a square matrix of size length x length"
    for i in range(len(prob_cs)):
        for j in range(len(prob_cs)):
            if i != j and dependence[i, j]:
                # if dependence from concept c_i to c_j, replace c_i with p(parity(c_i, c_j))
                # note that p(parity(c_i, c_j)) = p(c_i)*p(c_j) + (1-p(c_i))*(1-p(c_j))
                prob_cs[i] = prob_cs[i] * prob_cs[j] + (1 - prob_cs[i]) * (1 - prob_cs[j])
    return prob_cs


def compute_joint_c_given_x(c_vec: List[int], x: ndarray, p_noise: List[float], p_inds: List[List[int]],
                            dependence: Optional[List[List[int]]] = None) -> float:
    """
    Compute the probability of a concept vector (e.g., (1, 0, 1) given x (e.g., [0, 1, 1, 0, 1])
    :param dependence:
    :param c_vec:
    :param x:
    :param p_noise:
    :param p_inds:
    :return:
    """
    p_c_vec = 1.0
    p_cs = []
    for i, c in enumerate(c_vec):
        p_c = compute_c_i_proba(x, p_noise[i], p_inds[i])
        p_cs.append(p_c)
    if dependence is not None:
        p_cs = update_c_i_probas(p_cs, dependence)
    for i, c in enumerate(c_vec):
        if c == 1:
            p_c_vec *= p_cs[i]
        else:
            p_c_vec *= 1 - p_cs[i]
    return p_c_vec


def compute_concept(X: ndarray, p_noise: float, p_inds: List[int]) -> ndarray:
    """
    Features to concepts function, parity of bits in sets of 3
    :param X: features
    :param p_noise: probability of using parity function for concepts
    :param p_inds: indices from features for parity functions
    :return: ndarray of concepts
    """
    # c = np.array([np.random.randint(2) if bernoulli(concept_noise_probs) else parity(x[parity_inds]) for x in X])
    c = np.array([bernoulli(compute_c_i_proba(x, p_noise, p_inds)) for x in X])
    return c

def introduce_dependence(C_probs: List[ndarray], p_dependence: float) -> List[ndarray]:
    """
    Introduces dependence between concepts based on the given probability.

    :param C_probs: A list of concept probabilities arrays. The length of the list is n_concepts and the length of each
                    array is n_samples.
    :param p_dependence: The probability of dependence
    :return: A list of concept probabilities arrays with introduced dependence.
    """
    # Create a copy of the concept probabilities
    C_dep = C_probs.copy()

    if p_dependence > 0:
        # Sample n_samples from a Bernoulli RV
        D = [bernoulli(0.7) for _ in range(len(C_probs[0]))]

        # Iterate over each concept
        for c_i in range(len(C_dep)):
            # Introduce dependence by modifying the concept probabilities based on the random variables
            C_dep[c_i] = np.array([min(cp_i + (p_dependence * d_i), 1.0)
                                   for cp_i, d_i in zip(C_probs[c_i], D)])

    # Return the new list of concept probabilities with dependence introduced
    return C_dep


def compute_prob_y_given_c(c: List[int], coefs: List[float], intcpt: float) -> float:
    return logistic(sum([coef * c_i for coef, c_i in zip(coefs, c)]) + intcpt)


def compute_prob_ys_given_C(C: ndarray, coefs: List[float], intcpt: float) -> ndarray:
    n_samples = C.shape[0]
    return np.array(
        [compute_prob_y_given_c(c=C[i, :], coefs=coefs, intcpt=intcpt) for i in range(n_samples)])


def compute_prob_y_given_X(X: ndarray, coefs: List[float], intcpt: float, concept_noise_probs: List[float],
                           parity_inds: List[List[int]], dependence: Optional[List[List[int]]] = None) -> ndarray:
    n_samples = X.shape[0]
    n_concepts = len(concept_noise_probs)
    all_cs = generate_all_binary_vectors(n_concepts)
    prob = np.zeros(n_samples)
    for i in range(n_samples):
        total_c_probs = []
        for c in all_cs:
            p_c = compute_joint_c_given_x(c, X[i], concept_noise_probs, parity_inds, dependence)
            total_c_probs.append(p_c)
            p_y_given_c = compute_prob_y_given_c(c, coefs, intcpt)
            prob[i] += p_c * p_y_given_c
        assert_almost_equal(sum(total_c_probs), 1.0)
    return prob


def compute_y_given_X(X: ndarray, coefs: List[float], intcpt: float, concept_noise_probs: List[float],
                      parity_inds: List[List[int]], dependence: Optional[List[List[int]]] = None) -> ndarray:
    prob = compute_prob_y_given_X(X, coefs, intcpt, concept_noise_probs, parity_inds, dependence)
    return np.array([bernoulli(p) for p in prob])
