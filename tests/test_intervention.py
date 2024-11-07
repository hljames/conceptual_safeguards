import dill
import os
import numpy as np
from collections import defaultdict

import intervention
import saved_models
from saved_models import ConceptBottleneckModel

# Instantiate the model
model = ConceptBottleneckModel()
model.n_concepts = 3


class TestCBM(ConceptBottleneckModel):
    def __init__(self):
        super().__init__()
        self.probas_dict = defaultdict(lambda: 0.5)
        self.probas_dict.update({
            (0.5, 0.8, 0.7): 0.6,
            (1.0, 0.8, 0.7): 0.9,
            (0.0, 0.8, 0.7): 0.87,
            (0.5, 1.0, 0.7): 0.8,
            (0.5, 0.0, 0.7): 0.77,
            (0.5, 0.8, 1.0): 0.7,
            (0.5, 0.8, 0.0): 0.67,

            (0.8, 0.5, 0.7): 0.6,
            (1.0, 0.5, 0.7): 0.7,
            (0.0, 0.5, 0.7): 0.67,
            (0.8, 1.0, 0.7): 0.8,
            (0.8, 0.0, 0.7): 0.77,
            (0.8, 0.5, 1.0): 0.9,
            (0.8, 0.5, 0.0): 0.87,
        })

    def concept_probas_to_y_proba(self, concept_probas, **kwargs):
        n_samples, n_concepts = concept_probas.shape
        y_probas = np.zeros((n_samples,))
        for i in range(n_samples):
            y_probas[i] = self.probas_dict[tuple(concept_probas[i, :])]
        return y_probas

def test_variance_intervention():
    data_dir = '/Users/hailey/repos/uacbm/datasets/05_03_21_54/'
    dataset_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    with open(os.path.join(dataset_dirs[0], 'training.pkl'), 'rb') as f_train:
        training_set = dill.load(f_train)
    with open(os.path.join(dataset_dirs[0], 'validation.pkl'), 'rb') as f_train:
        validation_set = dill.load(f_train)
    m = saved_models
    all_models_dir = os.path.join(dataset_dirs[0], 'models')
    model_dirs = [f.path for f in os.scandir(all_models_dir) if f.is_dir()]
    for model_dir in model_dirs:
        if 'up_cbm' in model_dir:
            m = saved_models.PropagationCBM()
            m.load(model_dir)
    n_concepts = 3
    concept_probas = np.array([0.5, 0.8, 0.7]).reshape(1, n_concepts)
    i = k = 0
    k = 2
    cint_0 = concept_probas[i, :].copy().reshape(1, n_concepts)
    cint_1 = concept_probas[i, :].copy().reshape(1, n_concepts)
    cint_0[:, k], cint_1[:, k] = 0, 1
    f_cint_0 = m.concept_probas_to_y_proba(cint_0)[0]
    f_cint_1 = m.concept_probas_to_y_proba(cint_1)[0]
    p_k = concept_probas[i, k]
    E_r_k = p_k * f_cint_1 + (1 - p_k) * f_cint_0
    E_r_k_squared = p_k * (f_cint_1 ** 2) + (1 - p_k) * (f_cint_0 ** 2)
    score = E_r_k_squared - (E_r_k ** 2)
    print(score)


# def test_intervention_impact():
#     data_dir = '/Users/hailey/repos/uacbm/datasets/05_03_21_54/'
#     dataset_dirs = [f.path for f in os.scandir(data_dir) if f.is_dir()]
#     with open(os.path.join(dataset_dirs[0], 'training.pkl'), 'rb') as f_train:
#         training_set = dill.load(f_train)
#     with open(os.path.join(dataset_dirs[0], 'validation.pkl'), 'rb') as f_train:
#         validation_set = dill.load(f_train)
#     m = models
#     all_models_dir = os.path.join(dataset_dirs[0], 'models')
#     model_dirs = [f.path for f in os.scandir(all_models_dir) if f.is_dir()]
#     for model_dir in model_dirs:
#         if 'up_cbm' in model_dir:
#             m = models.PropagationCBM()
#             m.load(model_dir)
#     n_concepts = 3
#     concept_probas = np.array([0.5, 0.8, 0.7]).reshape(1, n_concepts)
#     i = k = 0
#     cint_0 = concept_probas[i, :].copy().reshape(1, n_concepts)
#     cint_1 = concept_probas[i, :].copy().reshape(1, n_concepts)
#     cint_0[:, k], cint_1[:, k] = 0, 1
#     f_cint_0 = m.concept_probas_to_y_proba(cint_0)[0]
#     f_cint_1 = m.concept_probas_to_y_proba(cint_1)[0]
#     f_cpred = m.concept_probas_to_y_proba(concept_probas[i, :].reshape(1, n_concepts))[0]
#     p_k = concept_probas[i, k]
#     score = (abs(f_cint_0 - 0.5) - abs(f_cpred - 0.5)) * (1 - p_k) + (
#             abs(f_cint_1 - 0.5) - abs(f_cpred - 0.5)) * p_k
#     assert score != 0, "Test intervention impact failed"


def test_intervene():
    C_pred = np.array([[0.5, 0.8, 0.7]])
    C_true = np.array([[1, 1, 1]])
    intervention_masks = np.array([[False, True, False]])
    C_pred_intervened = intervention.intervene(C_pred, C_true, intervention_masks)
    assert np.array_equal(C_pred_intervened, np.array([[0.5, 1.0, 0.7]])), "Test intervene failed"


def test_random_intervention():
    costs = [1., 1., 1.]
    f = TestCBM()
    h = saved_models.SelectiveClassifier(prediction_threshold=0.7)
    random_int = intervention.RandomIntervention(costs, f, h)
    C_pred = np.array([[0.5, 0.8, 0.7], [0.8, 0.5, 0.7]])
    random_int.fit(C_pred, budget=3)
    assert random_int.p_intervene == 0.5, "Test fit random intervention failed"


def test_expectation_intervention():
    costs = [1., 1., 1.]
    f = TestCBM()
    h = saved_models.SelectiveClassifier(prediction_threshold=0.7)
    ua_int = intervention.VarianceIntervention(costs, f, h)
    concept_probas = np.array([[0.5, 0.8, 0.7], [0.8, 0.5, 0.7]])
    ua_int.fit(concept_probas, budget=3)
    assert np.isclose(ua_int.score_threshold, 0.00018899999999999473), "Test fit expectation intervention failed"

# from intervention import prob_abstention
#
# def test_prob_abstention():
#     c_proba = np.array([0.2, 0.8, 0.7])
#     y_proba_intervened_0 = np.array([0.3, 0.6, 0.45])
#     y_proba_intervened_1 = np.array([0.8, 0.4, 0.7])
#     prediction_threshold = 0.65
#
#     expected_output = np.array([0.0, 1.0, 0.3])
#
#     assert np.allclose(
#         prob_abstention(c_proba, y_proba_intervened_0, y_proba_intervened_1, prediction_threshold),
#         expected_output,
#         atol=1e-6
#     )


# def test_create_masks_from_concept_importances():
#     concept_importances = np.array([0.2, 0.5, 0.1, 0.4])
#     n_interventions = 2
#     intervention_strategy = intervention.ImportantConceptsIntervention(n_interventions)
#     expected_mask = np.array([False, True, False, True])
#     mask = intervention_strategy.create_masks_from_concept_importances(concept_importances)
#     assert np.array_equal(mask, expected_mask), "Test 1 failed"
