import os

import dill
import numpy as np
from numpy.testing import assert_allclose

import saved_models
from saved_models import ConceptBottleneckModel #, intervene_on_concept, intervene

# Instantiate the model
model = ConceptBottleneckModel()
model.n_concepts = 3

# Define some inputs for testing
C_pred = np.array([[1, 0, 0], [0, 1, 0]])
C_pred = np.array([[1, 0, 0], [0, 1, 0]])
C = np.array([[0, 1, 1], [1, 0, 1]])
intervention_policy = [0.5, 0.3, 0.8]




def test_additive_interventions():
    """Test if E[f(\bar{c})}| \psi_{a}] + E[f(\bar{c})}| \psi_{b}] = E[f(\bar{c})}| \psi_{a} + \psi_{b}]"""
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
    psi_a = np.array([[False, False, True]])
    psi_b = np.array([[False, True, False]])
    psi_a_b = np.array([[False, True, True]])
    # for t in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    for t in [.85]:
            # C_pred_probas = np.array([[0.5, 0.99, 0.01]])
        # for C_pred_probas in [np.array([[0.5, 0.5, 0.5]]), np.array([[0.5, 0.3, 0.3]]), np.array([[0.5, 0.3, 0.9]]),
        #                       np.array([[0.5, 0.99, 0.99]]), np.array([[0.5, 0.01, 0.01]]),
        #                       np.array([[0.5, 0.45, 0.55]])]:
        for C_pred_probas in [np.array([[0.5, 0.45, 0.55]])]:
            y_proba = float(m.C_probas_to_y_proba(C_pred_probas))
            C_true_a1 = np.array([[np.nan, np.nan, 1.0]])
            C_true_a0 = np.array([[np.nan, np.nan, 0.0]])
            C_true_b1 = np.array([[np.nan, 1.0, np.nan]])
            C_true_b0 = np.array([[np.nan, 0.0, np.nan]])
            C_true_a1_b1 = np.array([[np.nan, 1.0, 1.0]])
            C_true_a1_b0 = np.array([[np.nan, 1.0, 0.0]])
            C_true_a0_b1 = np.array([[np.nan, 0.0, 1.0]])
            C_true_a0_b0 = np.array([[np.nan, 0.0, 0.0]])
            C_pred_probas_a1 = m.intervene(C_pred_probas, C_true_a1, intervention_masks=psi_a)
            C_pred_probas_a0 = m.intervene(C_pred_probas, C_true_a0, intervention_masks=psi_a)
            f_cbar_a1_imp = float(m.C_probas_to_y_proba(C_pred_probas_a1))
            f_cbar_a0_imp = float(m.C_probas_to_y_proba(C_pred_probas_a0))
            f_cbar_a1 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_a1, intervention_masks=psi_a))
            delta_certainty_a1 = min(1 - y_proba, y_proba) - min(1 - f_cbar_a1, f_cbar_a1)
            f_cbar_a0 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_a0, intervention_masks=psi_a))
            delta_certainty_a0 = min(1 - y_proba, y_proba) - min(1 - f_cbar_a0, f_cbar_a0)
            f_cbar_b1 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_b1, intervention_masks=psi_b))
            f_cbar_b0 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_b0, intervention_masks=psi_b))
            f_cbar_a1_b1 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_a1_b1, intervention_masks=psi_a_b))
            f_cbar_a1_b0 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_a1_b0, intervention_masks=psi_a_b))
            f_cbar_a0_b1 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_a0_b1, intervention_masks=psi_a_b))
            f_cbar_a0_b0 = float(m.C_probas_to_y_proba(C_pred_probas, C_true=C_true_a0_b0, intervention_masks=psi_a_b))

            prob_C_true_a1 = np.prod(
                np.where(np.isnan(C_true_a1), 1.0, np.where(C_true_a1, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_a0 = np.prod(
                np.where(np.isnan(C_true_a0), 1.0, np.where(C_true_a0, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_b1 = np.prod(
                np.where(np.isnan(C_true_b1), 1.0, np.where(C_true_b1, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_b0 = np.prod(
                np.where(np.isnan(C_true_b0), 1.0, np.where(C_true_b0, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_a1_b1 = np.prod(
                np.where(np.isnan(C_true_a1_b1), 1.0, np.where(C_true_a1_b1, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_a1_b0 = np.prod(
                np.where(np.isnan(C_true_a1_b0), 1.0, np.where(C_true_a1_b0, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_a0_b1 = np.prod(
                np.where(np.isnan(C_true_a0_b1), 1.0, np.where(C_true_a0_b1, C_pred_probas, 1 - C_pred_probas)))
            prob_C_true_a0_b0 = np.prod(
                np.where(np.isnan(C_true_a0_b0), 1.0, np.where(C_true_a0_b0, C_pred_probas, 1 - C_pred_probas)))

            E_fc_psi_a = prob_C_true_a1 * f_cbar_a1 + prob_C_true_a0 * f_cbar_a0
            E_fc_psi_b = prob_C_true_b1 * f_cbar_b1 + prob_C_true_b0 * f_cbar_b0
            E_fc_psi_a_b = prob_C_true_a1_b1 * f_cbar_a1_b1 + prob_C_true_a1_b0 * f_cbar_a1_b0 + prob_C_true_a0_b1 * f_cbar_a0_b1 + prob_C_true_a0_b0 * f_cbar_a0_b0
            E_F_psi_a = prob_C_true_a1 * (f_cbar_a1 > t or f_cbar_a1 < (1 - t)) + prob_C_true_a0 * (
                        f_cbar_a0 > t or f_cbar_a0 < (1 - t))
            E_F_psi_b = prob_C_true_b1 * (f_cbar_b1 > t or f_cbar_b1 < (1 - t)) + prob_C_true_b0 * (
                        f_cbar_b0 > t or f_cbar_b0 < (1 - t))
            E_F_psi_a_b = prob_C_true_a0_b0 * (f_cbar_a0_b0 > t or f_cbar_a0_b0 < (1 - t)) + prob_C_true_a0_b1 * (
                        f_cbar_a0_b1 > t or f_cbar_a0_b1 < (1 - t)) + prob_C_true_a1_b0 * (
                                      f_cbar_a1_b0 > t or f_cbar_a1_b0 < (1 - t)) + prob_C_true_a1_b1 * (
                                      f_cbar_a1_b1 > t or f_cbar_a1_b1 < (1 - t))
            print('t: ', t)
            print('concept_probas: ', C_pred_probas)
            print(' f_{c3=1}: ', round(f_cbar_a1, 3), ' f_{c3=0}: ', round(f_cbar_a0, 3), ' f_{c2=1}: ',
                  round(f_cbar_b1, 3), ' f_{c2=0}: ', round(f_cbar_b0, 3), ' f_{c2=1,c3=1}: ', round(f_cbar_a1_b1, 3),
                  ' f_{c2=0,c3=1}: ', round(f_cbar_a1_b0, 3), ' f_{c2=1,c3=0}: ', round(f_cbar_a0_b1, 3),
                  ' f_{c2=0,c3=0}: ', round(f_cbar_a0_b0, 3))
            print('y_proba: ', y_proba, ' E[f|\psi_3]: ', E_fc_psi_a, ' E[f|\psi_2]: ', E_fc_psi_b,
                  ' E[f|\psi_{2,3}]: ', E_fc_psi_a_b, '\n\n E[F|\psi_3]: ', E_F_psi_a, ' E[F|\psi_2]: ', E_F_psi_b,
                  ' E[F|\psi_{2,3}]: ', E_F_psi_a_b)
            print('\n\n\n')