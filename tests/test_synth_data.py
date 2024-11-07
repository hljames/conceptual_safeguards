"""
Install pytest and pytest-cov:
pip3 install pytest
pip3 install pytest-cov

To run all tests:
pytest

To run all tests and compute test coverage:
pytest --cov
"""

import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from data import synth_data as s


@pytest.fixture
def settings():
    return


def test_parity():
    # sum of digits is odd
    assert s.parity([1, 1, 1])
    assert not s.parity([0, 1, 1])


# def test_introduce_dependence():
#     C = [np.array([0, 1, 1, 0, 1]),
#          np.array([1, 0, 1, 0, 0]),
#          np.array([1, 1, 0, 0, 1])]
#     # dependence from c1 to c2
#     C_dep = s.introduce_dependence(C, dependence=[[0, 0, 0], [0, 0, 1], [0, 0, 0]])
#     # new c1 is 1 if sum of digits between previous c1 and c2 is odd and 0 otherwise
#     C_dep_expected = [np.array([0, 1, 1, 0, 1]),
#                       np.array([0, 1, 1, 0, 1]),
#                       np.array([1, 1, 0, 0, 1])]
#     assert_equal(C_dep, C_dep_expected)

# def test_introduce_dependence():
#     """not working"""
#     n_concepts, input_dim = 3, 5
#     p_inds = [[0, 1, 3], [0, 1, 2], [0, 1, 4]]
#     p_noise = [0.1, 0.2, 0.3]
#     dependence = [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
#     all_possible_cs = utils.generate_all_binary_vectors(n_concepts)
#     all_possible_xs = utils.generate_all_binary_vectors(input_dim)
#     joint_probabilities = np.zeros((len(all_possible_xs), len(all_possible_cs)))
#     marginal_probabilities = np.zeros((len(all_possible_xs), len(all_possible_cs)))
#     for i, x in enumerate(all_possible_xs):
#         x = np.asarray(x)
#         probas_c = [0.] * n_concepts
#         for c_i in range(n_concepts):
#             prob_c_i = s.compute_c_i_proba(x, p_noise[c_i], p_inds[c_i])
#             probas_c[c_i] = prob_c_i
#         if dependence is not None:
#             probas_c = s.update_c_i_probas(probas_c, dependence)
#         for j, c in enumerate(all_possible_cs):
#             joint_probabilities[i,j] = s.compute_joint_c_given_x(c, x, p_noise, p_inds, dependence)
#             marginal_p = 1.0
#             for c_ind, c_val in enumerate(c):
#                 if c_val == 1:
#                     marginal_p *= probas_c[c_ind]
#                 else:
#                     marginal_p *= 1 - probas_c[c_ind]
#             marginal_probabilities[i,j] = marginal_p
#     assert_raises(AssertionError, np.testing.assert_almost_equal, joint_probabilities, marginal_probabilities)


def test_compute_compute_c_i_proba():
    # if concept_noise_probs = 0, then concept is parity function
    assert s.compute_c_i_proba(x=np.array([0, 1, 0, 0, 1]), p_noise=0, p_inds=[0, 1, 2]) == 1.0
    assert s.compute_c_i_proba(x=np.array([0, 0, 0, 0, 1]), p_noise=0, p_inds=[0, 1, 2]) == 0.0

    # check that sum of probabilities is 1
    c_vecs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    total_p = 0
    x = np.array([0, 1, 0, 0, 1])
    p_inds = [[0, 1, 3], [0, 1, 2], [0, 1, 4]]
    p_noise = [0.1, 0.2, 0.3]
    for c_vec in c_vecs:
        p_c_vec = 1.0
        for i, c in enumerate(c_vec):
            p_c = s.compute_c_i_proba(x, p_noise[i], p_inds[i])
            if c == 1:
                p_c_vec *= p_c
            else:
                p_c_vec *= 1 - p_c
        total_p += p_c_vec
    assert_almost_equal(total_p, 1.0)
