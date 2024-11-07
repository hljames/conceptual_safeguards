"""
Model architectures
"""
import time
from typing import Optional, List, Dict, Tuple

import dill
import joblib
import numpy as np
import os
import shutil
import torch.multiprocessing
from prettytable import PrettyTable
from sklearn.metrics import f1_score

import utils

from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from constants import PROCESSED_DATASETS_DIR
from models.concept_model import load_concept_models
from data.data import CBMDataset, downsample_by_index
from intervention import intervene
from metrics import calculate_metrics, get_metric_str, error

torch.multiprocessing.set_sharing_strategy('file_system')


class SelectiveClassifier(object):
    """
    Selective classifier
    """

    def __init__(self, prediction_threshold: float = 0.5):
        self.prediction_threshold = prediction_threshold
        pass

    def get_abstension_mask(self, y_proba: ndarray, prediction_threshold=None) -> ndarray:
        if prediction_threshold is None:
            prediction_threshold = self.prediction_threshold
        if len(y_proba.shape) == 1:
            return (y_proba <= prediction_threshold) & (y_proba >= (1 - prediction_threshold))
        else:
            return np.amax(y_proba, axis=1) < prediction_threshold

    def predict_proba(self, y_proba: ndarray, prediction_threshold=None) -> ndarray:
        """
        predict labels given y_proba using threshold
        :param prediction_threshold:
        :param y_proba: ndarray
        :return: ndarray
        """
        if prediction_threshold is None:
            prediction_threshold = self.prediction_threshold
        y_proba = y_proba.copy()
        # y_pred_proba = np.empty_like(y_proba, dtype=float)
        # if len(y_pred_proba.shape) > 1:
        #     # replace y_pred_proba with 1 where y_proba > prediction_threshold (for each row)
        #     y_pred_proba[y_proba > prediction_threshold] = 1
        # else:
        #     y_pred_proba[y_proba > prediction_threshold] = 1
        #     y_pred_proba[y_proba < (1 - prediction_threshold)] = 0
        abstention_mask = self.get_abstension_mask(y_proba=y_proba, prediction_threshold=prediction_threshold)
        if len(y_proba.shape) == 1:
            y_proba[abstention_mask] = None
        else:
            y_proba[abstention_mask, :] = None
        return y_proba


class ConceptBottleneckModel:
    def __init__(self,
                 concept_models_dir: Optional[str] = '',
                 **kwargs):
        self.random_state = kwargs.get('random_state', 1)
        self.lr_solver = kwargs.get('lr_solver', 'lbfgs')
        self.lr_penalty = kwargs.get('lr_penalty', 'l2')
        self.multiclass = kwargs.get('multiclass', False)
        self.num_workers = kwargs.get('num_workers', 1)

        # submodels
        self.concept_models_dir = concept_models_dir
        self.concept_subset = kwargs.get('concept_subset', 'all')
        self.n_features = kwargs.get('n_features', 0)
        self._concept_models = []
        self.y_model = None
        self.labels = None  # y model labels for calculating metrics

        # prediction caching
        self.save_path = kwargs.get('save_path', None)
        self.use_cached_y_proba = kwargs.get('use_cached_y_proba', True)

        # y model calibration
        self.calibration_method = kwargs.get('calibration_method', 'sigmoid')
        self.y_model_precalibration = None
        self.cal_on_preds = kwargs.get('cal_on_preds', False)

        # # concept model calibration
        # self.calibrated_concept_models_dir = kwargs.get('calibrated_concept_models_dir', '')
        # if self.calibrated_concept_models_dir:
        #     self.calibrated_concept_models = load_concept_models(self.calibrated_concept_models_dir)
        #     assert len(self.calibrated_concept_models) == len(self.concept_models)
        # else:
        #     self.calibrated_concept_models = []

    @property
    def pred_proba_dir(self):
        return os.path.join(self.save_path, 'pred_proba/')

    @property
    def concept_models(self):
        if not self._concept_models:
            assert self.concept_models_dir, 'Concept models directory not specified'
            self._concept_models = load_concept_models(self.concept_models_dir)
        return self._concept_models

    @property
    def n_concepts(self):
        return len(self.concept_models)

    def __getstate__(self):
        state = {k: v for (k, v) in self.__dict__.items()}
        state.pop('_concept_models')
        state.pop('y_model')
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)

    def reset_cached_predictions(self):
        # delete self.pred_proba_dir directory
        if os.path.exists(self.pred_proba_dir):
            shutil.rmtree(self.pred_proba_dir)
        # create new self.pred_proba_dir directory
        os.makedirs(self.pred_proba_dir)

    def fit_from_paths(self,
                       training_set_path: str,
                       validation_set_path: str,
                       **kwargs):
        """
        Fit the model from paths to pickled datasets
        :param training_set_path:
        :param validation_set_path:
        :param kwargs:
        :return:
        """
        self.reset_cached_predictions()
        self.fit_y_model_from_paths(training_set_path, validation_set_path, **kwargs)

    def fit_y_model_from_paths(self, training_set_path: str, validation_set_path: str, **kwargs):
        """
        Fit the y model from paths to pickled datasets
        :param training_set_path:
        :param validation_set_path:
        :param kwargs:
        :return:
        """
        self.reset_cached_predictions()
        with open(training_set_path, 'rb') as train_file:
            training_set = dill.load(train_file)
        assert training_set.n_concepts == self.n_concepts
        self.labels = np.unique(training_set.y)
        multiclass = 'multinomial' if self.multiclass else 'auto'
        self.y_model = LogisticRegression(random_state=self.random_state, multi_class=multiclass, max_iter=1000,
                                          solver=self.lr_solver, penalty=self.lr_penalty)
        # print('Fitting y model...')
        if self.concept_subset == 'all':
            c_idx = list(range(training_set.n_concepts))
        else:
            c_idx = [c_i for c_i, c in enumerate(training_set.concept_names) if c in self.concept_subset]
        self.y_model.fit(training_set.C[:, c_idx], training_set.y)
        # print f1 score for true concepts
        print('F1 score for true concepts: ', f1_score(training_set.y, self.y_model.predict(training_set.C[:, c_idx]),
                                                       average='weighted'))
        print('Error for true concepts: ', error(training_set.y,
                                                 self.y_model.predict(training_set.C[:, c_idx])))

    def fit(self,
            training_set: CBMDataset,
            validation_set: CBMDataset,
            **kwargs):
        assert self.n_concepts == training_set.n_concepts
        self.labels = np.unique(training_set.y)
        self.fit_y_model(training_set, validation_set, **kwargs)

    def fit_y_model(self,
                    training_set: CBMDataset,
                    validation_set: CBMDataset,
                    **kwargs):
        self.reset_cached_predictions()
        multiclass = 'multinomial' if self.multiclass else 'auto'
        self.y_model = LogisticRegression(random_state=self.random_state, multi_class=multiclass, max_iter=1000,
                                          solver=self.lr_solver, penalty=self.lr_penalty)
        if self.concept_subset == 'all':
            c_idx = list(range(training_set.n_concepts))
        else:
            c_idx = [c_i for c_i, c in enumerate(training_set.concept_names) if c in self.concept_subset]
        self.y_model.fit(training_set.C[:, c_idx], training_set.y)
        # self.y_model.fit(training_set.C, training_set.y)

    def evaluate_dataset_from_path(self, dataset_path: str, **kwargs):
        with open(dataset_path, 'rb') as dataset_file:
            y_true = dill.load(dataset_file).y
        use_cached_concept_probas = kwargs.get('use_cached_concept_probas', None)
        y_pred_proba = self.predict_y_proba_from_path(dataset_path,
                                                      use_cached_concept_probas=use_cached_concept_probas)
        y_pred = self.predict_y_from_path(dataset_path, y_pred_proba,
                                          use_cached_concept_probas=use_cached_concept_probas)
        return calculate_metrics(y_true, y_pred, y_pred_proba, labels=self.labels)

    def save(self, save_dir, overwrite=False):
        if os.path.exists(save_dir):
            if not overwrite:
                raise ValueError('Directory already exists')
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'cbm'), 'wb') as outp:  # Overwrites any existing file.
            dill.dump(self.__getstate__(), outp, dill.HIGHEST_PROTOCOL)
        self.save_y_model(save_dir)

    def save_y_model(self, save_dir):
        joblib.dump(self.y_model, os.path.join(save_dir, 'y_model'))
        if self.y_model_precalibration is not None:
            joblib.dump(self.y_model_precalibration, os.path.join(save_dir, 'y_model_precalibration'))

    def load_y_model(self, load_dir):
        self.y_model = joblib.load(os.path.join(load_dir, 'y_model'))
        if os.path.exists(os.path.join(load_dir, 'y_model_precalibration')):
            self.y_model_precalibration = joblib.load(os.path.join(load_dir, 'y_model_precalibration'))

    def load(self, load_dir):
        with open(os.path.join(load_dir, 'cbm'), 'rb') as inp:
            self.__setstate__(dill.load(inp))
        self.load_y_model(load_dir)

    def predict_y(self,
                  dataset: CBMDataset,
                  y_proba: Optional[ndarray] = None,
                  intervention_masks: Optional[ndarray] = None) -> np.ndarray:
        if not y_proba:
            y_proba = self.predict_y_proba(dataset, intervention_masks)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred

    def predict_y_from_path(self,
                            dataset_path: str,
                            y_proba: Optional[ndarray] = None,
                            intervention_masks: Optional[ndarray] = None,
                            threshold: float = 0.5,
                            use_cached_concept_probas: Optional[bool] = None) -> np.ndarray:
        if y_proba is None:
            y_proba = self.predict_y_proba_from_path(dataset_path,
                                                     intervention_masks,
                                                     use_cached_concept_probas=use_cached_concept_probas)
        if self.multiclass:
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = (y_proba >= threshold).astype(int)
        return y_pred

    def concept_probas_to_y_proba(self,
                                  concept_probas: ndarray,
                                  dataset: Optional[CBMDataset] = None,
                                  intervention_masks: Optional[ndarray] = None):
        C_pred = self.concept_probas_to_concepts(concept_probas=concept_probas)
        if intervention_masks is not None and intervention_masks.any():
            assert intervention_masks.shape == C_pred.shape
            if self.concept_subset == 'all':
                c_idx = list(range(dataset.n_concepts))
            else:
                c_idx = [c_i for c_i, c in enumerate(dataset.concept_names) if c in self.concept_subset]
            C_pred = intervene(C_pred=C_pred, C_true=dataset.C[:, c_idx],
                               intervention_masks=intervention_masks)
        y_proba = self.y_model.predict_proba(C_pred)
        if not self.multiclass:
            y_proba = y_proba[:, 1]
        return y_proba

    def predict_y_proba(self,
                        dataset: CBMDataset,
                        intervention_masks: Optional[ndarray] = None) -> ndarray:
        concept_probas = self.predict_concept_probas(dataset)
        y_proba = self.concept_probas_to_y_proba(concept_probas, dataset, intervention_masks)
        return y_proba

    def predict_y_proba_from_path(self,
                                  dataset_path: str,
                                  intervention_masks: Optional[ndarray] = None,
                                  use_cached_y_proba: Optional[bool] = None,
                                  use_cached_concept_probas: Optional[bool] = None) -> ndarray:
        if use_cached_y_proba is None:
            use_cached_y_proba = self.use_cached_y_proba
        pred_proba_path = os.path.join(dataset_path.replace(PROCESSED_DATASETS_DIR, self.pred_proba_dir),
                                       'pred_proba.npy').replace('.pkl', '')
        os.makedirs(os.path.dirname(pred_proba_path), exist_ok=True)
        if use_cached_y_proba and os.path.exists(pred_proba_path):
            # print('Loading cached predictions from {}'.format(pred_proba_path))
            concept_probas = None
            y_proba = np.load(pred_proba_path)
        else:
            concept_probas = self.predict_concept_probas_from_path(dataset_path,
                                                                   use_cached_concept_probas=use_cached_concept_probas)
            y_proba = self.concept_probas_to_y_proba(concept_probas)
            np.save(pred_proba_path, y_proba)
        # replace predictions from intervention masks
        if intervention_masks is not None and intervention_masks.any():
            dataset = dill.load(open(dataset_path, 'rb'))
            # get indices of rows where any of the intervention masks is True
            if intervention_masks.ndim == 1:
                intervention_masks = intervention_masks.reshape(-1, 1)
            int_inds = np.where(intervention_masks.any(axis=1))[0]
            dataset_int_inds = downsample_by_index(dataset, int_inds)
            if concept_probas is None:
                concept_probas = self.predict_concept_probas_from_path(dataset_path,
                                                                       use_cached_concept_probas=use_cached_concept_probas)
            concept_probas_int = concept_probas[int_inds, :]
            y_proba_int = self.concept_probas_to_y_proba(concept_probas_int,
                                                         dataset=dataset_int_inds,
                                                         intervention_masks=intervention_masks[int_inds])
            y_proba[int_inds] = y_proba_int
        return y_proba

    def predict_concepts(self, dataset: CBMDataset, concept_threshold: float = 0.5) -> ndarray:
        concept_probas = self.predict_concept_probas(dataset)
        c_hat = self.concept_probas_to_concepts(concept_probas, concept_threshold=concept_threshold)
        return c_hat

    @staticmethod
    def concept_probas_to_concepts(concept_probas: ndarray, concept_threshold: float = 0.5, ) -> ndarray:
        return np.where(concept_probas >= concept_threshold, 1, 0)

    def predict_concept_proba(self, dataset: CBMDataset, c_i: int) -> ndarray:
        cm = self.concept_models[c_i]
        assert c_i == cm.concept_index
        return cm.predict_proba_dataset(dataset, dataset_as_is=True, num_workers=self.num_workers)

    def predict_concept_proba_from_path(self,
                                        dataset_path: str,
                                        c_i: int,
                                        use_cached_concept_probas: Optional[bool] = None) -> ndarray:
        cm = self.concept_models[c_i]
        assert c_i == cm.concept_index
        return cm.predict_proba_dataset_from_path(dataset_path, use_cached_predictions=use_cached_concept_probas,
                                                  dataset_as_is=True, num_workers=self.num_workers)

    def predict_concept_probas_from_path(self,
                                         dataset_path: str,
                                         use_cached_concept_probas: Optional[bool] = None) -> ndarray:
        # print(f'predicting concept probas from path {dataset_path}')
        with open(dataset_path, 'rb') as dataset_file:
            dataset = dill.load(dataset_file)
        if self.concept_subset == 'all':
            c_idx = list(range(dataset.n_concepts))
        else:
            c_idx = [c_i for c_i, c in enumerate(dataset.concept_names) if c in self.concept_subset]
        c_hat_proba = np.empty(shape=(dataset.n, len(c_idx)))
        for i, c_i in enumerate(c_idx):
            c_i_proba = self.predict_concept_proba_from_path(dataset_path, c_i,
                                                             use_cached_concept_probas=use_cached_concept_probas)
            try:
                c_hat_proba[:, i] = c_i_proba
            except:
                assert 0
        return c_hat_proba

    def predict_concept_probas(self, dataset: CBMDataset) -> ndarray:
        if self.concept_subset == 'all':
            c_idx = list(range(dataset.n_concepts))
        else:
            c_idx = [c_i for c_i, c in enumerate(dataset.concept_names) if c in self.concept_subset]
        c_hat_proba = np.empty(shape=(dataset.n, len(c_idx)))
        for i, c_i in enumerate(c_idx):
            c_i_proba = self.predict_concept_proba(dataset, c_i)
            c_hat_proba[:, i] = c_i_proba
        return c_hat_proba

    def get_concept_importances(self):
        return self.y_model.coef_[0]

    def calibrate_y_model_from_path(self,
                                    calibration_set_path: str,
                                    use_cached_concept_probas: Optional[bool] = None):
        print(f'calibrating y model from path using {self.calibration_method}')
        self.reset_cached_predictions()
        self.y_model_precalibration = self.y_model
        assert self.y_model_precalibration is not None, 'y model must be trained before calibration'
        calibration_set = dill.load(open(calibration_set_path, 'rb'))
        C_true = calibration_set.C
        if self.cal_on_preds:
            C_pred_proba = self.predict_concept_probas_from_path(calibration_set_path,
                                                                 use_cached_concept_probas=use_cached_concept_probas)
            C_cal = self.concept_probas_to_concepts(C_pred_proba)
        else:
            C_cal = calibration_set.C
        print('STATS BEFORE CALIBRATION')
        y_pred_proba = self.y_model.predict_proba(C_true)
        if not self.multiclass:
            y_pred_proba = y_pred_proba[:, 1]
        stats_true = calculate_metrics(y_true=calibration_set.y, y_pred=self.y_model.predict(C_true),
                                        y_pred_proba=y_pred_proba, labels=self.labels)
        print('brier true: ', stats_true['brier'], )
        print('ece true: ', stats_true['ece'])
        stats_full = self.evaluate_dataset_from_path(calibration_set_path)
        print('brier full: ', stats_full['brier'])
        print('ece full: ', stats_full['ece'])
        model_cal = CalibratedClassifierCV(self.y_model_precalibration, method=self.calibration_method, cv="prefit")
        model_cal = model_cal.fit(C_cal, calibration_set.y.astype(np.float32))
        self.y_model = model_cal
        self.reset_cached_predictions()
        print('STATS AFTER CALIBRATION')
        y_pred_proba = self.y_model.predict_proba(C_true)
        if not self.multiclass:
            y_pred_proba = y_pred_proba[:, 1]
        stats_true_after = calculate_metrics(y_true=calibration_set.y, y_pred=self.y_model.predict(C_true),
                                  y_pred_proba=y_pred_proba, labels=self.labels)
        print('brier true: ', stats_true_after['brier'])
        print('ece true: ', stats_true_after['ece'])
        stats_full_after = self.evaluate_dataset_from_path(calibration_set_path)
        print('brier full: ', stats_full_after['brier'])
        print('ece full: ', stats_full_after['ece'])
        return self.y_model

    def calibrate_y_model(self, calibration_set):
        self.reset_cached_predictions()
        print(f'calibrating y model using {self.calibration_method}')
        self.y_model_precalibration = self.y_model
        model_cal = CalibratedClassifierCV(estimator=self.y_model, method=self.calibration_method, cv="prefit")
        if self.cal_on_preds:
            C_pred_proba = self.predict_concept_probas(calibration_set)
            C_cal = self.concept_probas_to_concepts(C_pred_proba)
        else:
            C_cal = calibration_set.C
        model_cal.fit(C_cal, calibration_set.y.astype(np.float32))
        self.y_model = model_cal


class SequentialCBM(ConceptBottleneckModel):
    def __init__(self,
                 concept_models_dir: Optional[str] = '',
                 **kwargs):
        super().__init__(concept_models_dir, **kwargs)
        self.str_desc = 'sequential'

    def fit_y_model_from_paths(self,
                               training_set_path: str,
                               validation_set_path: str,
                               use_cached_concept_probas: Optional[bool] = None,
                               **kwargs):
        """
        Fit on predicted probabilities from the concept models
        :param training_set_path:
        :param validation_set_path:
        :param use_cached_concept_probas:
        :param kwargs:
        :return:
        """
        self.reset_cached_predictions()
        train_C = self.predict_concept_probas_from_path(training_set_path,
                                                        use_cached_concept_probas=use_cached_concept_probas)
        training_set = dill.load(open(training_set_path, 'rb'))
        multiclass = 'multinomial' if self.multiclass else 'auto'
        self.labels = np.unique(training_set.y)
        self.y_model = LogisticRegression(random_state=self.random_state, multi_class=multiclass, max_iter=1000,
                                          solver=self.lr_solver, penalty=self.lr_penalty)
        print('using solver: ', self.lr_solver)
        self.y_model.fit(train_C, training_set.y)

    def fit_y_model(self,
                    training_set: CBMDataset,
                    validation_set: CBMDataset,
                    **kwargs):
        self.reset_cached_predictions()
        # fit on the predicted probabilities from the concept models
        train_C = self.predict_concept_probas(training_set)
        multiclass = 'multinomial' if self.multiclass else 'auto'
        self.labels = np.unique(training_set.y)
        self.y_model = LogisticRegression(random_state=self.random_state, multi_class=multiclass, max_iter=1000,
                                          solver=self.lr_solver, penalty=self.lr_penalty)
        print('using solver: ', self.lr_solver)
        self.y_model.fit(train_C, training_set.y)

    def calibrate_y_model_from_path(self,
                                    calibration_set_path: str,
                                    use_cached_concept_probas: Optional[bool] = None):
        """
        Calibrate the y model on the calibration set
        :param calibration_set_path:
        :param use_cached_concept_probas:
        :return:
        """
        self.reset_cached_predictions()
        self.y_model_precalibration = self.y_model
        model_cal = CalibratedClassifierCV(self.y_model, method=self.calibration_method, cv="prefit")
        calibration_set = dill.load(open(calibration_set_path, 'rb'))
        if self.cal_on_preds:
            C_pred_proba = self.predict_concept_probas_from_path(calibration_set_path,
                                                                 use_cached_concept_probas=use_cached_concept_probas)
            C_cal = self.concept_probas_to_concepts(C_pred_proba)
        else:
            C_cal = calibration_set.C
        model_cal.fit(C_cal, calibration_set.y.astype(np.float32))
        self.y_model = model_cal

    def calibrate_y_model(self, calibration_set: CBMDataset):
        self.reset_cached_predictions()
        self.y_model_precalibration = self.y_model
        model_cal = CalibratedClassifierCV(self.y_model, method=self.calibration_method, cv="prefit")
        if self.cal_on_preds:
            C_pred_proba = self.predict_concept_probas(calibration_set)
            C_cal = self.concept_probas_to_concepts(C_pred_proba)
        else:
            C_cal = calibration_set.C
        model_cal.fit(C_cal, calibration_set.y.astype(np.float32))
        self.reset_cached_predictions()
        self.y_model = model_cal


class IndependentCBM(ConceptBottleneckModel):
    def __init__(self,
                 concept_models_dir: Optional[str] = '',
                 **kwargs):
        super().__init__(concept_models_dir, **kwargs)
        self.str_desc = 'indepen'


class PropagationCBM(ConceptBottleneckModel):
    def __init__(self,
                 concept_models_dir: Optional[str] = '',
                 **kwargs):
        super().__init__(concept_models_dir, **kwargs)
        self.use_fast_propagation = kwargs.get('use_fast_propagation', None)
        # if self.use_fast_propagation:
        #     print('Using Monte Carlo sampling for uncertainty propagation')
        self.str_desc = 'propagation'
        self.all_cs = []
        self.all_cs_probas = {}

    def update_all_cs_probas(self):
        if self.use_fast_propagation:
            return
        if not self.all_cs:
            print('getting all cs...')
            # all possible concept combinations (e.g., [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1] . . .)
            self.all_cs = utils.generate_all_binary_vectors(self.n_concepts)
        print('getting pred probas for all cs...')
        pred_probas = self.y_model.predict_proba(np.asarray(self.all_cs).reshape(-1, 1))
        if not self.multiclass:
            pred_probas = pred_probas[:, 1]
        self.all_cs_probas = {tuple(c): proba for c, proba in zip(self.all_cs, pred_probas)}

    @staticmethod
    def c_probability(c: list, c_pred_proba: ndarray):
        n_samples = c_pred_proba.shape[0]
        prod = np.ones(shape=(n_samples,))
        # get probs for concept that are 0 and 1
        prod_one = c_pred_proba[:, np.array(c) == 1]
        prod_zero = c_pred_proba[:, np.array(c) == 0]
        # Compute product for weighting
        # if prod_one.any():
        prod *= np.prod(prod_one, axis=1)
        # if prod_zero.any():
        prod *= np.prod(1 - prod_zero, axis=1)
        return prod

    @staticmethod
    def propagation_monte_carlo(concept_probas: ndarray, y_model, n_mc: int = 1000, multiclass: bool = False):
        """
        Monte Carlo sampling for uncertainty propagation
        :param concept_probas:
        :param y_model:
        :param n_mc:
        :param multiclass:
        :return:
        """
        # start_time = time.time()
        # n_samples, n_concepts = concept_probas.shape[0], concept_probas.shape[1]
        # y_pred_proba = []
        # for c_proba in concept_probas:
        #     cs = (np.random.rand(n_mc, n_concepts) < c_proba).astype(int)
        #     y_proba = np.mean(y_model.predict_proba(cs), axis=0)
        #     y_pred_proba.append(y_proba)
        # if not multiclass:
        #     y_pred_proba = np.asarray(y_pred_proba)[:, 1]
        # print(f'Propagation took {time.time() - start_time} seconds')
        # return np.asarray(y_pred_proba)

        start_time = time.time()
        n_samples, n_concepts = concept_probas.shape
        # Generate the entire 3D random array at once (n_samples x n_mc x n_concepts)
        random_values = np.random.rand(n_samples, n_mc, n_concepts)
        # Compare with concept_probas to generate cs for all samples and Monte Carlo runs at once
        cs = (random_values < concept_probas[:, np.newaxis, :]).astype(int)
        # Reshape to 2D array to predict all at once, then reshape back to separate the samples
        predicted_probas = y_model.predict_proba(cs.reshape(-1, n_concepts)).reshape(n_samples, n_mc, -1)
        # Calculate the mean prediction for each sample
        y_pred_proba = np.mean(predicted_probas, axis=1)
        if not multiclass:
            y_pred_proba = y_pred_proba[:, 1]
        # print(f'Propagation took {time.time() - start_time} seconds')
        return y_pred_proba

    @staticmethod
    def propagation_logreg_decomposition(concept_probas: ndarray, y_model):
        weights = y_model.coef_[0]
        bias = y_model.intercept_[0]

        # Calculate the decomposed version of f_hat
        n_concepts = concept_probas.shape[1]
        bias_per_feature = bias / n_concepts

        # Vectorized function to compute individual contributions
        def f_k_vectorized(c_k, w_k, b_k):
            return 1 / (1 + np.exp(-w_k * c_k - b_k))

        # Calculate average f_k across all samples and features
        avg_f_k_values = np.mean(f_k_vectorized(concept_probas, weights, bias_per_feature), axis=1)
        return avg_f_k_values

    @staticmethod
    def propagation(concept_probas: ndarray,
                    all_cs: List[List[int]],
                    all_cs_probas: Dict[Tuple[int], ndarray],
                    multiclass: bool = False) -> ndarray:
        concept_probas = concept_probas.copy()
        n_samples = len(concept_probas)
        outer_sum = np.zeros(shape=(n_samples,))
        for c_i, c in enumerate(all_cs):
            prod = PropagationCBM.c_probability(c, concept_probas)
            # get y_pred proba for c_vec tiled across n_samples
            y_pred_proba = np.tile(all_cs_probas[tuple(c)], (n_samples,))
            if multiclass:
                # get the probability of the most likely class
                f_c = np.max(y_pred_proba, axis=1).reshape(-1, )
            else:
                # get the probability of the positive class
                f_c = y_pred_proba
            prod *= f_c
            outer_sum += prod
        return outer_sum

    def concept_probas_to_y_proba(self,
                                  concept_probas: ndarray,
                                  dataset: Optional[CBMDataset] = None,
                                  intervention_masks: Optional[ndarray] = None) -> ndarray:
        if intervention_masks is not None and intervention_masks.any():
            if self.concept_subset == 'all':
                c_idx = list(range(dataset.n_concepts))
            else:
                c_idx = [c_i for c_i, c in enumerate(dataset.concept_names) if c in self.concept_subset]
            concept_probas = intervene(C_pred=concept_probas, C_true=dataset.C[:, c_idx],
                                       intervention_masks=intervention_masks)
        if self.use_fast_propagation:
            return PropagationCBM.propagation_monte_carlo(concept_probas, self.y_model,
                                                          multiclass=self.multiclass)
        else:
            if not self.all_cs:
                # all possible concept combinations (e.g., [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1] . . .)
                self.all_cs = utils.generate_all_binary_vectors(self.n_concepts)
            if not self.all_cs_probas:
                self.update_all_cs_probas()
            return self.propagation(concept_probas, self.all_cs, self.all_cs_probas, multiclass=self.multiclass)

    def fit_y_model(self,
                    training_set: CBMDataset,
                    validation_set: CBMDataset,
                    **kwargs):
        super().fit_y_model(training_set, validation_set, **kwargs)
        if self.use_fast_propagation is None:
            self.use_fast_propagation = training_set.n_concepts > 5
            # if self.use_fast_propagation:
            # print('Using Monte Carlo sampling for uncertainty propagation')
        self.update_all_cs_probas()

    def fit_y_model_from_paths(self, training_set_path: str, validation_set_path: str, **kwargs):
        super().fit_y_model_from_paths(training_set_path, validation_set_path, **kwargs)
        if self.use_fast_propagation is None:
            with open(training_set_path, 'rb') as f:
                training_set = dill.load(f)
            self.use_fast_propagation = training_set.n_concepts > 5
            # if self.use_fast_propagation:
            #     print('Using Monte Carlo sampling for uncertainty propagation')
        self.update_all_cs_probas()

    def calibrate_y_model(self, calibration_set):
        pass


# class InterventionAwareCBM(ConceptBottleneckModel):
#     """
#     Concept bottleneck model is trained using a fixed intervention policy with p(intervention) < 1
#     """
#
#     def __init__(self,
#                  concept_models_dir: str,
#                  **kwargs):
#         super().__init__(concept_models_dir, **kwargs)
#         # intervetion policy model pairs
#         self.y_models = {}
#         self.intervention_policy = None
#
#     def fit_y_model(self,
#                     training_set: CBMDataset,
#                     validation_set: CBMDataset,
#                     **kwargs) -> LogisticRegression:
#         intervention_policy = kwargs.get('intervention_policy', None)
#         self.intervention_policy = intervention_policy
#         save_path = kwargs.get('save_path', '')
#         save_path = os.path.join(save_path, f'ym_{self.intervention_policy}')
#         if save_path in self.y_models:
#             self.y_model = self.y_models[save_path]
#             return self.y_model
#         # get concept predictions
#         C_pred_train = self.predict_concepts(training_set)
#         # impute interventions
#         if intervention_policy is not None:
#             # generate random probabilities for each training example
#             rand_floats = np.random.rand(len(training_set), self.n_concepts)
#             # set True values based on the intervention_policy probabilities
#             intervention_masks = rand_floats < np.array(intervention_policy)
#             C_pred_train = intervene(C_pred=C_pred_train, C_true=training_set.C,
#                                      intervention_masks=intervention_masks)
#         # fit y model
#         self.y_model = LogisticRegression(random_state=self.random_state).fit(C_pred_train, training_set.y)
#         self.y_models[save_path] = self.y_model
#         return self.y_model
#
#
# class DependentConceptsBaseline(ConceptBottleneckModel):
#     def __init__(self,
#                  concept_models_dir: str,
#                  **kwargs):
#         super().__init__(concept_models_dir, **kwargs)
#         self.all_cs = []
#
#     def concept_probas_to_y_proba(self,
#                                   concept_probas: ndarray,
#                                   dataset: Optional[CBMDataset] = None,
#                                   intervention_masks: Optional[ndarray] = None,
#                                   threshold: float = 0.5):
#         # all possible concept combinations (e.g., [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1] . . .)
#         if not self.all_cs:
#             self.all_cs = utils.generate_all_binary_vectors(self.n_concepts)
#         outer_sum = torch.zeros(size=(len(concept_probas), 1), requires_grad=False)
#         # predict probabilities for each concept for each sample
#         assert concept_probas.shape == (len(concept_probas), self.n_concepts)
#         for c_vec in self.all_cs:
#             # nsamples x 1
#             prod = torch.ones(size=(len(concept_probas), 1), requires_grad=False)
#             # for each of the possible concept combinations, multiply the probabilities of each concept
#             for c_i, c in enumerate(c_vec):
#                 c_proba = concept_probas[:, c_i].reshape(-1, 1)
#                 if intervention_masks is not None:
#                     # replace probabilities with 1.0 or 0.0 if the concept is intervened on
#                     c_proba = intervene_on_concept(
#                         c_pred=c_proba.squeeze(),
#                         c_true=dataset.C[:, c_i],
#                         intervention_mask=intervention_masks[:, c_i]
#                     ).reshape(-1, 1)
#                 if c == 1:
#                     prod *= c_proba
#                 elif c == 0:
#                     prod *= (1 - c_proba)
#             c_vec_tile = np.tile(c_vec, [len(concept_probas), 1])
#             f_c = self.y_model.predict_proba(c_vec_tile)[:, 1].reshape(-1, 1)
#             prod *= f_c
#             outer_sum += prod
#         return outer_sum.float().detach().squeeze().numpy()


def train_concept_bottleneck_models(train_file_path: str,
                                    validation_file_path: str,
                                    model_types: List[str],
                                    cbms_save_dir: str,
                                    concept_models_dir: str,
                                    # calibrated_concept_models_dir: Optional[str] = None,
                                    retrain_models: bool = False,
                                    print_stats: bool = False,
                                    test_file_path: Optional[str] = None,
                                    **kwargs):
    os.makedirs(cbms_save_dir, exist_ok=True)
    use_cached_concept_probas = kwargs.get('use_cached_concept_probas', True)
    use_cached_y_proba = kwargs.get('use_cached_y_proba', True)
    kwargs['use_cached_concept_probas'] = use_cached_concept_probas
    kwargs['use_cached_y_proba'] = use_cached_y_proba
    results_dict = {}
    for model_type in model_types:
        print(f"Model type: {model_type}")
        if model_type == 'indepen':
            m = IndependentCBM(**kwargs)
        elif model_type == 'cal_indepen':
            m = IndependentCBM(**kwargs)
            m.str_desc = m.str_desc.replace('indepen', 'cal_indepen')
        elif model_type == 'propagation':
            m = PropagationCBM(**kwargs)
        elif model_type == 'sequential':
            m = SequentialCBM(**kwargs)
        else:
            raise ValueError(f"Unknown cbm model type {model_type}")
        cbm_model_path = os.path.join(cbms_save_dir, m.str_desc)
        m.save_path = cbm_model_path
        if retrain_models or not os.path.exists(os.path.join(cbm_model_path, 'cbm')):
            m.concept_models_dir = concept_models_dir
            m.fit_from_paths(train_file_path, validation_file_path, **kwargs)
            if kwargs.get('calibrate_y_model', False) or model_type == 'cal_indepen':
                m.calibrate_y_model_from_path(calibration_set_path=validation_file_path,
                                              use_cached_concept_probas=use_cached_concept_probas)
            m.save(cbm_model_path, overwrite=True)
        m.load(cbm_model_path)
        print('random state: ', m.random_state)
        # m.save(cbm_model_path, overwrite=True)
        # m.load(cbm_model_path)
        assert m.save_path == cbm_model_path
        if print_stats:
            # train_metrics_dict = m.evaluate_dataset_from_path(train_file_path, **kwargs)
            # print(f'train_log_loss: {get_metric_str(train_metrics_dict, "log_loss")} '
            #       f'train_err: {get_metric_str(train_metrics_dict, "error")} '
            #       f'train_ece: {get_metric_str(train_metrics_dict, "ece")}',
            #       f'train_f1: {get_metric_str(train_metrics_dict, "f1")}',
            #       f'train_auc_roc: {get_metric_str(train_metrics_dict, "auc_roc")}',
            #       )
            val_metrics_dict = m.evaluate_dataset_from_path(validation_file_path, **kwargs)
            print(f'valid_log_loss: {get_metric_str(val_metrics_dict, "log_loss")} '
                  f'valid_err: {get_metric_str(val_metrics_dict, "error")} '
                  f'valid_ece: {get_metric_str(val_metrics_dict, "ece")}',
                  f'valid_f1: {get_metric_str(val_metrics_dict, "f1")}',
                  f'valid_auc_roc: {get_metric_str(val_metrics_dict, "auc_roc")}',
                  )
            test_metrics_dict = None
            if test_file_path is not None:
                test_metrics_dict = m.evaluate_dataset_from_path(test_file_path, **kwargs)
                print(f'test_log_loss: {get_metric_str(test_metrics_dict, "log_loss")} '
                      f'test_err: {get_metric_str(test_metrics_dict, "error")} '
                      f'test_ece: {get_metric_str(test_metrics_dict, "ece")}',
                      f'test_f1: {get_metric_str(test_metrics_dict, "f1")}',
                      f'test_auc_roc: {get_metric_str(test_metrics_dict, "auc_roc")}',
                      )

            results_dict[model_type] = {
                # 'Training': train_metrics_dict,
                'Validation': val_metrics_dict,
                'Test': test_metrics_dict
            }
    if print_stats:
        table = PrettyTable()
        # table.field_names = ["Model",
        #                      "N Train", "N VAl", "N Test",
        #                      "Train % Pos", "Val % Pos", "Test % Pos",
        #                      "Train Loss", "Val Loss", "Test Loss",
        #                      "Train Err", "Val Err", "Test Err",
        #                      "Train ECE", "Val ECE", "Test ECE",
        #                      "Train F1", "Val F1", "Test F1",
        #                      "Train AUC", "Val AUC", "Test AUC",
        #                      "Train Brier", "Val Brier", "Test Brier"]
        table.field_names = ["Model",
                             "N VAl", "N Test",
                             "Val % Pos", "Test % Pos",
                             "Val Loss", "Test Loss",
                             "Val Err", "Test Err",
                             "Val ECE", "Test ECE",
                             "Val F1", "Test F1",
                             "Val AUC", "Test AUC",
                             "Val Brier", "Test Brier"]

        for model_type, metrics in results_dict.items():
            row = [model_type]

            for metric_name in ["n_samples", "percent_positives", "log_loss", "error", "ece", "f1", "auc_roc", "brier"]:
                # for dataset in ['Training', 'Validation', 'Test']:
                for dataset in ['Validation', 'Test']:
                    metrics_dataset = metrics.get(dataset, {})
                    if metric_name == 'n_samples':
                        metric_value = get_metric_str(metrics_dataset, metric_name, digits=0)
                    else:
                        metric_value = get_metric_str(metrics_dataset, metric_name)
                    row.append(metric_value)

            table.add_row(row)
        print(table)
    print('saved models to: ', cbms_save_dir)
