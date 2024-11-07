import shutil
from collections import defaultdict

import dill
import torch
import skorch
import os
import time

from joblib import dump as dump_joblib
from joblib import load as load_joblib
from prettytable import PrettyTable
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.dummy import DummyClassifier
from torchmetrics.classification import BinaryF1Score
from scipy.special import expit as sigmoid
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skorch import NeuralNetBinaryClassifier
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from typing import Tuple, List

from data.data import CBMDataset, downsample_by_index, downsample_to_balance_concept, downsample_by_concept
from constants import EMBEDDING_MODEL_DIR, PROCESSED_DATASETS_DIR
from data.image_data import ImageCBMDataset
from metrics import *
from models.embedding_model import embed_concept_dataset, embed_concept_dataset_path, get_model


class CustomDummyClassifier(DummyClassifier):
    def predict_proba(self, X):
        print('predicting proba custom dummy classifier')
        proba = super().predict_proba(X)
        print('predicting proba custom dummy classifier')
        extended_proba = np.zeros((proba.shape[0], 2))
        extended_proba[:, 0] = proba[:, 0]
        extended_proba[:, 1:] = 1 - extended_proba[:, 0].reshape(-1, 1)
        return extended_proba


class ConceptModel(object):
    """
    Concept detector interface
    """

    def __init__(self, **kwargs):
        self.model = None
        self.random_state = kwargs.get('random_state', 1)
        # save path for loading caches predictions
        self.save_path = kwargs.get('save_path', None)
        # required for training on CBMDataset
        self.concept_index = kwargs.get('concept_index', None)
        self.y_labels = kwargs.get('y_labels', None)
        # data setup
        self.calibration_method = kwargs.get('calibration_method', 'sigmoid')
        self.model_precalibration = None
        self.balance_by_concept = kwargs.get('balance_by_concept', False)
        self.max_samples = kwargs.get('max_samples', None)
        self.input_dim = kwargs.get('input_dim', 0)
        self.cm_device = kwargs.get('device', 'mps')
        self.batch_size = kwargs.get('batch_size', 64)
        self.num_workers = kwargs.get('num_workers', os.cpu_count() - 1)
        self.persistent_workers = kwargs.get('persistent_workers', True)
        # concept info
        self.concept_name = kwargs.get('concept_name', 'concept')
        self.concept_noise_prob = kwargs.get('concept_noise_prob', 0.0)
        # embedding model setup
        self.embedding_out_dir = kwargs.get('embedding_out_dir', EMBEDDING_MODEL_DIR)
        self.embedding_backbone_name = kwargs.get('embedding_backbone_name', None)
        self.use_cached_predictions = kwargs.get('use_cached_predictions', True)
        self.preprocess = kwargs.get('preprocess', None)

    @property
    def str_desc(self):
        str_desc = f'cm{self.concept_index}_{self.concept_name}'
        if self.concept_noise_prob > 0:
            str_desc += f'_noise{self.concept_noise_prob}'
        return str_desc

    @property
    def balance_concept_index(self):
        return self.concept_index if self.balance_by_concept else None

    @property
    def pred_proba_dir(self):
        return os.path.join(self.save_path, 'pred_proba/')

    def calibrate_model(self, calibration_set, embed_dataset=True, **kwargs):
        print('calibrating model')
        self.reset_cached_predictions()
        self.model_precalibration = self.model
        assert self.model_precalibration is not None, 'model must be trained before calibration'
        if self.concept_index is None:
            raise ValueError('concept_index is None')
        if embed_dataset:
            calibration_set = embed_concept_dataset(calibration_set,
                                                    embedding_backbone_name=self.embedding_backbone_name,
                                                    balance_concept_index=self.balance_concept_index,
                                                    max_samples=self.max_samples,
                                                    batch_size=self.batch_size,
                                                    num_workers=self.num_workers,
                                                    device=self.cm_device)
        print('STATS BEFORE CALIBRATION')
        print('brier: ', self.evaluate_dataset(calibration_set, embed_dataset=False)['brier'])
        print('ece: ', self.evaluate_dataset(calibration_set, embed_dataset=False)['ece'])
        model_cal = CalibratedClassifierCV(self.model_precalibration, method=self.calibration_method, cv="prefit")
        model_cal = model_cal.fit(calibration_set.X, calibration_set.C[:, self.concept_index].astype(np.float32))
        self.model = model_cal
        self.reset_cached_predictions()
        print('STATS AFTER CALIBRATION')
        print('brier: ', self.evaluate_dataset(calibration_set, embed_dataset=False)['brier'])
        print('ece: ', self.evaluate_dataset(calibration_set, embed_dataset=False)['ece'])
        return self.model

    def calibrate_from_path(self,
                            calibration_set_path: str):
        self.reset_cached_predictions()
        self.model_precalibration = self.model
        assert self.model_precalibration is not None, 'model must be trained before calibration'
        calibration_set = dill.load(open(calibration_set_path, 'rb'))
        model_cal = CalibratedClassifierCV(self.model_precalibration, method='sigmoid', cv="prefit")
        model_cal = model_cal.fit(calibration_set.X, calibration_set.C[:, self.concept_index].astype(np.float32))
        self.model = model_cal
        self.reset_cached_predictions()
        return self.model

    def fit(self, training_set: CBMDataset, validation_set: CBMDataset,
            embed_dataset: bool = False, **kwargs):
        """
        fit concept detector
        :param training_set:
        :param validation_set:
        :param embed_dataset:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def evaluate_dataset(self,
                         dataset: CBMDataset,
                         embed_dataset: bool = True,
                         idx: Optional[List[int]] = None,
                         threshold: float = 0.5) -> dict:
        """
        evaluate concept detector on dataset
        :param dataset:
        :param embed_dataset:
        :param idx:
        :param threshold:
        :return:
        """
        if not idx:
            # default to evaluating on all samples
            idx = list(range(len(dataset)))
        c_pred_proba = self.predict_proba_dataset(dataset, embed_dataset=embed_dataset, idx=idx)
        c_true = dataset.C[idx, self.concept_index]
        c_pred = np.where(c_pred_proba >= threshold, 1, 0)
        # c_pred_proba_2d = np.column_stack((1 - c_pred_proba, c_pred_proba))
        return calculate_metrics(c_true, c_pred, c_pred_proba)

    def evaluate_dataset_from_path(self,
                                   dataset_path: str,
                                   idx: Optional[List[int]] = None,
                                   threshold: float = 0.5,
                                   use_cached_predictions: Optional[bool] = None,
                                   dataset_as_is=False) -> dict:
        assert self.concept_index is not None, 'concept index must be set'
        balance_concept_index = self.balance_concept_index if not dataset_as_is else False
        max_samples = self.max_samples if not dataset_as_is else None
        dataset = embed_concept_dataset_path(dataset_path,
                                             embedding_backbone_name=self.embedding_backbone_name,
                                             balance_concept_index=balance_concept_index,
                                             max_samples=max_samples,
                                             batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             device=self.cm_device)
        # downsample to remove unknown concept labels
        # dataset = downsample_by_concept(dataset, self.concept_index)
        # return self.evaluate_dataset(dataset, embed_dataset=False, idx=idx, threshold=threshold)
        if not idx:
            # default to evaluating on all samples
            idx = list(range(len(dataset)))
        c_pred_proba = self.predict_proba_dataset_from_path(dataset_path,
                                                            idx=idx,
                                                            use_cached_predictions=use_cached_predictions)
        c_true = dataset.C[idx, self.concept_index]
        c_pred = np.where(c_pred_proba >= threshold, 1, 0)
        idx = np.where(np.isin(c_true, [0, 1]))[0]
        # c_pred_proba_2d = np.column_stack((1 - c_pred_proba, c_pred_proba))
        return calculate_metrics(c_true[idx], c_pred[idx], c_pred_proba[idx], labels=np.array([0, 1]))

    def predict_proba(self, X_input):
        """
        predict concept probabilities
        :param X_input:
        :return:
        """
        raise NotImplementedError

    def predict_proba_dataset(self,
                              dataset: CBMDataset,
                              embed_dataset: bool = True,
                              idx: Optional[List[int]] = None,
                              dataset_as_is=False,
                              **kwargs) -> ndarray:
        self.num_workers = kwargs.get('num_workers', self.num_workers)
        self.cm_device = kwargs.get('cm_device', self.cm_device)
        if not idx:
            # default to predicting on all samples
            idx = list(range(len(dataset)))
        dataset = downsample_by_index(dataset, idx)

        if embed_dataset:
            balance_concept_index = self.balance_concept_index if not dataset_as_is else False
            max_samples = self.max_samples if not dataset_as_is else None
            dataset = embed_concept_dataset(dataset,
                                            embedding_backbone_name=self.embedding_backbone_name,
                                            balance_concept_index=balance_concept_index,
                                            max_samples=max_samples,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            device=self.cm_device)
        if isinstance(dataset, ImageCBMDataset):
            # use dataloader to get predict_probas
            dataset.preprocess = self.preprocess
            dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
            c_hat_probas = []
            for X_input, _, _ in dataloader:
                c_hat_probas.append(self.predict_proba(X_input))
            c_hat_proba = np.concatenate(c_hat_probas)
            return c_hat_proba
        else:
            X_input, _, _ = dataset[:]
            return self.predict_proba(X_input)

    def predict(self, X_input, threshold: float = 0.5) -> ndarray:
        """
        predict concept from X_input
        :param X_input:
        :param threshold:
        :return:
        """
        c_hat_proba = self.predict_proba(X_input)
        c_hat = np.where(c_hat_proba >= threshold, 1, 0)
        return c_hat

    def predict_dataset(self,
                        dataset: CBMDataset,
                        c_hat_proba: Optional[ndarray] = None,
                        embed_dataset: bool = True,
                        idx: Optional[List[int]] = None,
                        threshold: float = 0.5) -> ndarray:
        """
        predict concept from dataset
        :param dataset:
        :param c_hat_proba:
        :param embed_dataset:
        :param idx:
        :param threshold:
        :return:
        """
        if c_hat_proba is None:
            c_hat_proba = self.predict_proba_dataset(dataset=dataset,
                                                     embed_dataset=embed_dataset,
                                                     idx=idx)
        c_hat = np.where(c_hat_proba >= threshold, 1, 0)
        return c_hat

    def predict_dataset_from_path(self,
                                  dataset_path: str,
                                  c_hat_proba: Optional[ndarray] = None,
                                  threshold: float = 0.5,
                                  idx: Optional[List[int]] = None,
                                  use_cached_predictions: Optional[bool] = None,
                                  dataset_as_is=False,
                                  **kwargs) -> ndarray:
        self.cm_device = kwargs.get('cm_device', self.cm_device)
        self.num_workers = kwargs.get('num_workers', self.num_workers)
        if c_hat_proba is None:
            c_hat_proba = self.predict_proba_dataset_from_path(dataset_path,
                                                               idx=idx,
                                                               use_cached_predictions=use_cached_predictions,
                                                               dataset_as_is=dataset_as_is)
        c_hat = np.where(c_hat_proba >= threshold, 1, 0)
        return c_hat

    def reset_cached_predictions(self):
        # delete self.pred_proba_dir directory
        if os.path.exists(self.pred_proba_dir):
            shutil.rmtree(self.pred_proba_dir)
        # create new self.pred_proba_dir directory
        os.makedirs(self.pred_proba_dir)

    def predict_proba_dataset_from_path(self,
                                        dataset_path: str,
                                        idx: Optional[List[int]] = None,
                                        use_cached_predictions: Optional[bool] = None,
                                        dataset_as_is=False,
                                        **kwargs) -> ndarray:
        self.num_workers = kwargs.get('num_workers', self.num_workers)
        self.cm_device = kwargs.get('cm_device', self.cm_device)
        if use_cached_predictions is None:
            use_cached_predictions = self.use_cached_predictions
        pred_proba_path = os.path.join(dataset_path.replace(PROCESSED_DATASETS_DIR, self.pred_proba_dir),
                                       'pred_proba.npy')
        pred_proba_path = pred_proba_path.replace('.pkl', '')
        if dataset_as_is:
            pred_proba_path = pred_proba_path.replace('.npy', '_as_is.npy')
        os.makedirs(os.path.dirname(pred_proba_path), exist_ok=True)
        # print('pred_proba_path: ', pred_proba_path)
        if use_cached_predictions and os.path.exists(pred_proba_path):
            # print('loading pred_proba from: ', pred_proba_path)
            c_hat_proba = np.load(pred_proba_path)
        else:
            balance_concept_index = self.balance_concept_index if not dataset_as_is else False
            max_samples = self.max_samples if not dataset_as_is else None
            dataset = embed_concept_dataset_path(dataset_path,
                                                 embedding_backbone_name=self.embedding_backbone_name,
                                                 balance_concept_index=balance_concept_index,
                                                 max_samples=max_samples,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 device=self.cm_device)
            c_hat_proba = self.predict_proba_dataset(dataset, embed_dataset=False, idx=idx)
            # print('saving pred_proba to: ', pred_proba_path)
            np.save(pred_proba_path, c_hat_proba)
        return c_hat_proba

    def fit_from_paths(self,
                       training_set_path: str,
                       validation_set_path: str,
                       **kwargs):
        """
        fit concept detector from paths
        :param training_set_path:
        :param validation_set_path:
        :param kwargs:
        :return:
        """
        self.reset_cached_predictions()
        training_set = embed_concept_dataset_path(training_set_path,
                                                  embedding_backbone_name=self.embedding_backbone_name,
                                                  balance_concept_index=self.balance_concept_index,
                                                  max_samples=self.max_samples,
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers,
                                                  device=self.cm_device)
        validation_set = embed_concept_dataset_path(validation_set_path,
                                                    embedding_backbone_name=self.embedding_backbone_name,
                                                    balance_concept_index=self.balance_concept_index,
                                                    max_samples=self.max_samples,
                                                    batch_size=self.batch_size,
                                                    num_workers=self.num_workers,
                                                    device=self.cm_device)
        self.fit(training_set, validation_set, embed_dataset=False, **kwargs)

    def save(self, concept_model_dir: str):
        """
        Save concept model inside concept_model_dir
        :param concept_model_dir:
        :return:
        """
        raise NotImplementedError

    def load(self, load_dir: str):
        """
        Load concept model to from directory
        :param load_dir:
        :return:
        """
        raise NotImplementedError


class MLPConceptModel(LightningModule, ConceptModel):
    """
    Multi-layer perceptron concept detector
    """

    def __init__(self, **kwargs):
        super().__init__()
        ConceptModel.__init__(self, **kwargs)
        # self.str_desc = self.str_desc + '_typemlp'

        # logging
        self.train_outputs = []
        self.val_outputs = []

        # training parameters
        self.lr = kwargs.get('lr', 1e-4)
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.min_delta = kwargs.get('min_delta', 0.0)
        self.patience = kwargs.get('patience', 5)

        # model
        self.l1 = kwargs.get('l1', 100)
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.l1),
            nn.ReLU(),
            nn.Linear(self.l1, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()

    @property
    def str_desc(self):
        str_desc = super().str_desc
        str_desc += '_typemlp'
        return str_desc

    def update_model(self, input_dim: Optional[int] = None, l1: Optional[int] = None):
        """
        update model
        :param input_dim:
        :param l1:
        :return:
        """
        self.input_dim = input_dim if input_dim is not None else self.input_dim
        self.l1 = l1 if l1 is not None else self.l1
        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.l1),
            nn.ReLU(),
            nn.Linear(self.l1, 1)
        )

    def predict_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx=None, dataloader_idx=0) -> torch.Tensor:
        return self(batch[0])  # , batch[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        compute concept given x
        :param x: Tensor
        :return: c Tensor
        """
        return self.layers(x)

    def on_train_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.train_outputs = []
        return

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx=None) -> float:
        _, loss, err, f1 = self._get_preds_loss_error_f1(batch)
        self.log('train_loss', loss)
        self.log('train_error', err, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_f1', f1, prog_bar=True, on_step=False, on_epoch=True)
        self.train_outputs.append({'loss': loss, 'error': err, 'f1': f1})
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean()
        avg_error = torch.stack([x['error'] for x in self.train_outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in self.train_outputs]).mean()
        # print(f'Epoch {self.current_epoch} - avg_train_loss: {avg_loss}, avg_train_f1: {avg_f1}')

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_outputs = []
        return

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx=None) -> torch.Tensor:
        """
        get validation metrics for logging (check during each epoch)
        :param batch:
        :param batch_idx:
        :return:
        """
        preds, loss, err, f1 = self._get_preds_loss_error_f1(batch)
        self.log('val_loss', loss)
        self.log('val_error', err, prog_bar=False)
        self.log('val_f1', f1, prog_bar=False)
        self.val_outputs.append({'loss': loss, 'error': err, 'f1': f1})
        return preds

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack([x['loss'] for x in self.val_outputs]).mean()
        avg_error = torch.stack([x['error'] for x in self.val_outputs]).mean()
        avg_f1 = torch.stack([x['f1'] for x in self.val_outputs]).mean()
        # print(f'Epoch {self.current_epoch} - avg_valid_loss: {avg_loss}, avg_valid_f1: {avg_f1}')

    def configure_optimizers(self):
        """
        define optimizer
        :return: e.g., Adam
        """
        return Adam(self.parameters(), lr=self.lr)

    def _get_preds_loss_error_f1(self, batch):
        """
        get preds, loss, acc for a certain batch (auxiliary function)
        :param batch:
        :return:
        """
        if self.concept_index is None:
            raise ValueError('concept_index is None')
        x, C, _ = batch
        c = C[:, self.concept_index]
        x = x.to(self.cm_device)
        logits = self(x)
        preds = logits > 0
        c = c.view(-1, 1)
        loss = self.loss(logits, c.float())
        acc = accuracy(preds, c, task='binary')
        err = (c != preds).float().mean()
        binary_f1_score = BinaryF1Score().to(self.cm_device)
        f1 = binary_f1_score(preds, c)
        return preds, loss, err, f1

    def predict_proba(self, X_input) -> ndarray:
        """
        predict concept probabilities
        :param X_input:
        :return:
        """
        self.eval()
        self.float()
        self.to(self.cm_device)
        X_input = X_input.to(self.cm_device)
        return sigmoid(self(X_input).cpu().detach()).squeeze().numpy()

    def fit(self, training_set: CBMDataset, validation_set: CBMDataset,
            embed_dataset: bool = False, **kwargs):
        """
        fit concept detector
        :param training_set:
        :param validation_set:
        :param embed_dataset:
        :param kwargs:
        :return:
        """
        self.reset_cached_predictions()
        if embed_dataset:
            training_set = embed_concept_dataset(training_set,
                                                 embedding_backbone_name=self.embedding_backbone_name,
                                                 balance_concept_index=self.balance_concept_index,
                                                 max_samples=self.max_samples,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 device=self.cm_device)
            validation_set = embed_concept_dataset(validation_set,
                                                   embedding_backbone_name=self.embedding_backbone_name,
                                                   balance_concept_index=self.balance_concept_index,
                                                   max_samples=self.max_samples,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   device=self.cm_device)
        print(f'training on {len(training_set)} samples on device {self.cm_device} with {self.num_workers} workers')
        # training_set = self.embed_concept_dataset(training_set)
        # validation_set = self.embed_concept_dataset(validation_set)

        # update input size
        self.update_model(input_dim=training_set.X[0].shape[0])

        # update descriptive parameters
        self.concept_index = kwargs.get('concept_index', self.concept_index)
        self.concept_name = kwargs.get('concept_name', self.concept_name)
        self.concept_noise_prob = kwargs.get('concept_noise_prob', self.concept_noise_prob)

        # update training parameters
        self.lr = kwargs.get('lr', self.lr)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.max_epochs = kwargs.get('max_epochs', self.max_epochs)
        self.min_delta = kwargs.get('min_delta', self.min_delta)
        self.patience = kwargs.get('patience', self.patience)
        self.num_workers = kwargs.get('num_workers', self.num_workers)

        # downsample to remove unknown concept values
        training_set = downsample_by_concept(training_set, self.concept_index)
        validation_set = downsample_by_concept(validation_set, self.concept_index)
        # downsample training set to balance concept if applicable
        if self.balance_by_concept:
            training_set = downsample_to_balance_concept(training_set, self.concept_index,
                                                         self.max_samples)
            validation_set = downsample_to_balance_concept(validation_set, self.concept_index,
                                                           self.max_samples)

        # train
        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                                  pin_memory=True)
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                                pin_memory=True)
        self.to(self.cm_device)
        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=True,
            mode='max'
        )
        trainer = Trainer(callbacks=[early_stop_callback], max_epochs=self.max_epochs)  # , progress_bar_refresh_rate=0)
        trainer.fit(self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return self

    def save(self, concept_model_dir: str, overwrite: bool = False):
        # Ensure the directory exists
        save_dir = os.path.join(concept_model_dir, self.str_desc)
        if not overwrite and os.path.exists(save_dir):
            raise FileExistsError(f"File {save_dir} already exists. Set overwrite=True to overwrite.")
        os.makedirs(save_dir, exist_ok=True)
        # save torch parameters
        torch_state_dict_save_path = os.path.join(save_dir, "torch_state_dict.pth")
        torch.save(self.state_dict(), torch_state_dict_save_path)
        # save model parameters
        model_state_save_path = os.path.join(save_dir, "model_state.pkl")
        with open(model_state_save_path, 'wb') as outp:  # Overwrites any existing file.
            state = {k: v for (k, v) in self.__dict__.items() if not k.startswith('_')}
            dill.dump(state, outp, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_dir: str):
        # todo move to super class (up to load torch parameters)
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Directory {load_dir} does not exist.")
        model_instance = cls()
        # Load model parameters
        model_state_load_path = os.path.join(load_dir, "model_state.pkl")
        with open(model_state_load_path, 'rb') as inp:
            state = dill.load(inp)
            model_instance.__setstate__(state)
            model_instance.update_model()
        # Load torch parameters
        torch_state_dict_load_path = os.path.join(load_dir, "torch_state_dict.pth")
        model_instance.load_state_dict(torch.load(torch_state_dict_load_path))
        return model_instance


class FineTunedLightningModule(LightningModule):
    def __init__(self, model, loss, optimizer, concept_index):
        super().__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.concept_index = concept_index

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, C, _ = batch
        c = C[:, self.concept_index]
        x.to(self.cm_device)
        c.to(self.cm_device)
        outputs, aux_outputs = self.model(x)
        loss = self.loss(outputs.squeeze(), c.float()) + 0.4 * self.loss(aux_outputs.squeeze(), c.float())
        preds = outputs > 0
        train_f1 = f1_score(c.cpu().numpy(), preds.cpu().numpy(), average='micro')
        self.log('train_loss', loss)
        self.log('train_f1', train_f1.astype('float32'))
        return {'train_loss': loss, 'train_f1': train_f1, 'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, C, _ = batch
        c = C[:, self.concept_index]
        outputs = self.model(x)
        # loss = self.loss(outputs.squeeze(), c.float()) + 0.4 * self.loss(aux_outputs.squeeze(), c.float())
        preds = outputs > 0
        val_f1 = f1_score(c.cpu().numpy(), preds.cpu().numpy(), average='micro')
        # self.log('val_loss', loss)
        self.log('val_f1', val_f1.astype('float32'))
        return {'val_f1': val_f1}

    def configure_optimizers(self):
        return self.optimizer


class FinetunedConceptModel(ConceptModel):
    """
    Concept model that finetunes a backbone model
    """

    def __init__(self, **kwargs):
        super().__init__()
        ConceptModel.__init__(self, **kwargs)
        self.embedding_backbone_name = None

        # model training
        self.lr = kwargs.get('lr', 1e-3)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.min_delta = kwargs.get('min_delta', 0.0)
        self.patience = kwargs.get('patience', 10)
        self.momentum = kwargs.get('momentum', 0.9)
        self.loss = nn.BCEWithLogitsLoss()
        self.pretrained_model_name = kwargs.get('pretrained_model_name', 'ham10000_inception')
        self.optimizer = None

    @property
    def str_desc(self):
        str_desc = super().str_desc
        str_desc += '_typefinetuned' + f'_{self.pretrained_model_name}'
        return str_desc

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        return state

    def __setstate__(self, d):
        self.__dict__.update(d)

    def update_model(self):
        if self.pretrained_model_name == 'ham10000_inception':
            # Convert named_parameters to a list and get the last N layers
            params = list(self.model.named_parameters())
            trainable_layers = 5  # Replace with the number of layers you want to keep trainable
            last_n_layers = params[-trainable_layers:]
            for name, param in params:
                param.requires_grad = False
            for name, param in last_n_layers:
                param.requires_grad = True
            self.model.fc = torch.nn.Linear(2048, 1)
            self.model.AuxLogits.fc = torch.nn.Linear(768, 1)
            for name, param in self.model.fc.named_parameters():
                param.requires_grad = True
            for name, param in self.model.AuxLogits.fc.named_parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError('pretrained model {} not implemented'.format(self.pretrained_model_name))

    def fit(self, training_set: CBMDataset, validation_set: CBMDataset, **kwargs):
        self.reset_cached_predictions()
        # downsample to remove unknown concept values
        training_set = downsample_by_concept(training_set, self.concept_index)
        validation_set = downsample_by_concept(validation_set, self.concept_index)
        # downsample training set to balance concept if applicable
        if self.balance_by_concept:
            training_set = downsample_to_balance_concept(training_set, self.concept_index,
                                                         self.max_samples)
            validation_set = downsample_to_balance_concept(validation_set, self.concept_index,
                                                           self.max_samples)

        print('training on {} samples'.format(len(training_set)))

        train_loader = DataLoader(training_set, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                                  pin_memory=True)
        val_loader = DataLoader(validation_set, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                                pin_memory=True)
        self.model, _, self.preprocess = get_model(backbone_name=self.pretrained_model_name,
                                                   device=self.cm_device, full_model=True)
        training_set.preprocess = self.preprocess
        validation_set.preprocess = self.preprocess
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.update_model()
        self.model.to(self.cm_device)
        lightning_model = FineTunedLightningModule(self.model, self.loss, self.optimizer, self.concept_index)
        early_stop_callback = EarlyStopping(
            monitor='val_f1',
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=True,
            mode='max'
        )
        trainer = Trainer(callbacks=[early_stop_callback], max_epochs=self.max_epochs)  # , progress_bar_refresh_rate=0)
        trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        self.model = lightning_model.model
        print("Finished fine-tuning")
        return self.model

    def fit_from_paths(self, training_set_path: str, validation_set_path: str, **kwargs):
        self.reset_cached_predictions()
        with open(training_set_path, 'rb') as f:
            training_set = dill.load(f)
        with open(validation_set_path, 'rb') as f:
            validation_set = dill.load(f)
        return self.fit(training_set, validation_set, **kwargs)

    def predict_proba(self, X_input) -> ndarray:
        """
        predict concept probabilities
        :param X_input:
        :return:
        """
        self.model.eval()
        self.model.float()
        self.model.to(self.cm_device)
        X_input = X_input.to(self.cm_device)
        return sigmoid(self.model(X_input).cpu().detach()).squeeze().numpy()

    def save(self, concept_model_dir: str, overwrite: bool = False):
        # Ensure the directory exists
        save_dir = os.path.join(concept_model_dir, self.str_desc)
        if not overwrite and os.path.exists(save_dir):
            raise FileExistsError(f"File {save_dir} already exists. Set overwrite=True to overwrite.")
        os.makedirs(save_dir, exist_ok=True)
        # save torch parameters
        torch_state_dict_save_path = os.path.join(save_dir, "torch_state_dict.pth")
        torch.save(self.model.state_dict(), torch_state_dict_save_path)
        # save model parameters
        model_state_save_path = os.path.join(save_dir, "model_state.pkl")
        with open(model_state_save_path, 'wb') as outp:  # Overwrites any existing file.
            state = {k: v for (k, v) in self.__dict__.items() if not k.startswith('_')}
            dill.dump(state, outp, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_dir: str):
        # todo move to super class (up to load torch parameters)
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Directory {load_dir} does not exist.")
        model_instance = cls()
        # Load model parameters
        model_state_load_path = os.path.join(load_dir, "model_state.pkl")
        with open(model_state_load_path, 'rb') as inp:
            state = dill.load(inp)
            model_instance.__setstate__(state)
        # Load torch parameters
        model_instance.update_model()
        torch_state_dict_load_path = os.path.join(load_dir, "torch_state_dict.pth")
        model_instance.model.load_state_dict(torch.load(torch_state_dict_load_path))
        return model_instance


class LogRegConceptModel(ConceptModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_iters = kwargs.get('max_iters', 100)
        self.solver = kwargs.get('logreg_solver', 'lbfgs')
        self._penalty = kwargs.get('logreg_penalty', 'none')
        self.calibrate_logreg = kwargs.get('calibrate_logreg', False)

        # model
        self.model = LogisticRegression(random_state=self.random_state, max_iter=self.max_iters, solver=self.solver,
                                        penalty=self.penalty, n_jobs=self.num_workers)

    @property
    def penalty(self):
        return None if self._penalty.lower() == 'none' else self._penalty

    @property
    def str_desc(self):
        str_desc = super().str_desc
        str_desc += '_typelogreg'
        return str_desc

    def __setstate__(self, d):
        self.__dict__.update(d)

    def fit(self, training_set: CBMDataset, validation_set: CBMDataset,
            embed_dataset: bool = False, **kwargs):
        """
        Fit the model to the training set.
        :param training_set:
        :param validation_set:
        :param embed_dataset:
        :param kwargs:
        :return:
        """
        if embed_dataset:
            training_set = embed_concept_dataset(training_set,
                                                 embedding_backbone_name=self.embedding_backbone_name,
                                                 balance_concept_index=self.balance_concept_index,
                                                 max_samples=self.max_samples,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 device=self.cm_device)
            validation_set = embed_concept_dataset(validation_set,
                                                   embedding_backbone_name=self.embedding_backbone_name,
                                                   balance_concept_index=self.balance_concept_index,
                                                   max_samples=self.max_samples,
                                                   batch_size=self.batch_size,
                                                   num_workers=self.num_workers,
                                                   device=self.cm_device)
        # update descriptive info
        self.concept_index = kwargs.get('concept_index', self.concept_index)
        if self.concept_index is None:
            raise ValueError('concept_index is None')
        training_set = downsample_by_concept(training_set, self.concept_index)
        train_str = f'training on {len(training_set)} samples, max iters {self.max_iters}, penalty {self.penalty}, ' \
                    f'solver {self.solver} with {self.num_workers} workers'
        if self.calibrate_logreg:
            train_str += f', calibration method {self.calibration_method}'
        print(train_str)
        self.concept_name = kwargs.get('concept_name', 'concept')
        self.concept_noise_prob = kwargs.get('concept_noise_prob', 0.0)
        train_C = training_set.C[:, self.concept_index]
        # check if train_C is constant values
        if np.all(train_C == train_C[0]):
            print('concept is constant, skipping training')
            self.model = CustomDummyClassifier(strategy='constant', constant=train_C[0])
            self.model.fit(training_set.X, train_C)
        else:
            self.model.fit(training_set.X, train_C)
            if self.calibrate_logreg:
                self.model = self.calibrate_model(validation_set, embed_dataset=False)

    def predict_proba(self, X_input):
        return self.model.predict_proba(X_input)[:, 1]

    def save(self, concept_model_dir: str, overwrite: bool = False):
        # Ensure the directory exists
        save_dir = os.path.join(concept_model_dir, self.str_desc)
        if not overwrite and os.path.exists(save_dir):
            raise FileExistsError(f"File {save_dir} already exists. Set overwrite=True to overwrite.")
        os.makedirs(save_dir, exist_ok=True)
        # save torch parameters
        sklearn_save_path = os.path.join(save_dir, "sklearn.joblib")
        dump_joblib(self.model, sklearn_save_path)
        # save model parameters
        model_state_save_path = os.path.join(save_dir, "model_state.pkl")
        with open(model_state_save_path, 'wb') as outp:  # Overwrites any existing file.
            state = {k: v for (k, v) in self.__dict__.items() if not k.startswith('_')}
            dill.dump(state, outp, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_dir: str):
        # todo put in super class
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Directory {load_dir} does not exist.")
        model_instance = cls()  # You might need to supply initial parameters to the constructor
        # Load model parameters
        model_state_load_path = os.path.join(load_dir, "model_state.pkl")
        with open(model_state_load_path, 'rb') as inp:
            state = dill.load(inp)
            model_instance.__setstate__(state)
        # Load sklearn model
        sklearn_load_path = os.path.join(load_dir, "sklearn.joblib")
        model_instance.model = load_joblib(sklearn_load_path)  # Load model parameters
        return model_instance


class LinearSVMConceptModel(ConceptModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # training parameters
        self.C = kwargs.get('SVC_C', 1.0)
        self.gamma = kwargs.get('gamma', 'scale')
        self.calibrate_svm = kwargs.get('calibrate_svm', False)

        # model
        self.model = SVC(kernel='linear', C=self.C, gamma=self.gamma, probability=self.calibrate_svm)

    @property
    def str_desc(self):
        str_desc = super().str_desc
        str_desc += '_typelinearsvm'
        return str_desc

    def __setstate__(self, d):
        self.__dict__.update(d)

    def fit(self, training_set: CBMDataset, validation_set: CBMDataset,
            embed_dataset: bool = False, **kwargs):
        """
        Fit the model to the training set.
        :param training_set:
        :param validation_set:
        :param embed_dataset:
        :param kwargs:
        :return:
        """
        self.reset_cached_predictions()
        if embed_dataset:
            training_set = embed_concept_dataset(training_set,
                                                 embedding_backbone_name=self.embedding_backbone_name,
                                                 balance_concept_index=self.balance_concept_index,
                                                 max_samples=self.max_samples,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 device=self.cm_device)
        # update descriptive info
        self.concept_index = kwargs.get('concept_index', self.concept_index)
        self.calibrate_svm = kwargs.get('calibrate_svm', self.calibrate_svm)
        self.gamma = kwargs.get('gamma', self.gamma)
        self.C = kwargs.get('C', self.C)
        if self.concept_index is None:
            raise ValueError('concept_index is None')
        training_set = downsample_by_concept(training_set, self.concept_index)
        print('training on {} samples'.format(len(training_set)))
        self.concept_name = kwargs.get('concept_name', 'concept')
        self.concept_noise_prob = kwargs.get('concept_noise_prob', 0.0)
        # self.str_desc = f'cm{self.concept_index}_{self.concept_name}'
        # if self.concept_noise_prob > 0:
        #     self.str_desc += f'_noise{self.concept_noise_prob}'
        # self.str_desc += f'_typelinearsvm'
        # train
        self.model.fit(training_set.X, training_set.C[:, self.concept_index])

    def predict_proba(self, X_input):
        return self.model.predict_proba(X_input)[:, 1]

    def save(self, concept_model_dir: str, overwrite: bool = False):
        # Ensure the directory exists
        save_dir = os.path.join(concept_model_dir, self.str_desc)
        if not overwrite and os.path.exists(save_dir):
            raise FileExistsError(f"File {save_dir} already exists. Set overwrite=True to overwrite.")
        os.makedirs(save_dir, exist_ok=True)
        # save torch parameters
        sklearn_save_path = os.path.join(save_dir, "sklearn.joblib")
        dump_joblib(self.model, sklearn_save_path)
        # save model parameters
        model_state_save_path = os.path.join(save_dir, "model_state.pkl")
        with open(model_state_save_path, 'wb') as outp:  # Overwrites any existing file.
            state = {k: v for (k, v) in self.__dict__.items() if not k.startswith('_')}
            dill.dump(state, outp, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_dir: str):
        # todo put in super class
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Directory {load_dir} does not exist.")
        model_instance = cls()  # You might need to supply initial parameters to the constructor
        # Load model parameters
        model_state_load_path = os.path.join(load_dir, "model_state.pkl")
        with open(model_state_load_path, 'rb') as inp:
            state = dill.load(inp)
            model_instance.__setstate__(state)
        # Load sklearn model
        sklearn_load_path = os.path.join(load_dir, "sklearn.joblib")
        model_instance.model = load_joblib(sklearn_load_path)  # Load model parameters
        return model_instance


class MLPConceptModelCalibrated(ConceptModel):
    """
    Calibrated MLPConceptModel
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_mlp = kwargs.get('base_mlp', None)
        self.max_iters = kwargs.get('max_iters', 100)
        self.model = None

    @property
    def str_desc(self):
        str_desc = super().str_desc
        str_desc += '_typemlp'
        return str_desc

    def __setstate__(self, d):
        self.__dict__.update(d)

    def fit(self, training_set: CBMDataset, validation_set: CBMDataset, **kwargs):
        if self.base_mlp is None:
            self.base_mlp = MLPConceptModel(**kwargs)
        self.base_mlp.concept_index = self.concept_index
        self.base_mlp.save_path = self.save_path
        self.base_mlp.cm_device = self.cm_device
        self.base_mlp.to(self.cm_device)
        self.base_mlp = self.base_mlp.fit(training_set, validation_set, **kwargs)
        self.model = NeuralNetBinaryClassifier(
            self.base_mlp,
            criterion=self.base_mlp.loss,
            optimizer=torch.optim.Adam,
            lr=self.base_mlp.lr,
            max_epochs=self.max_iters,
            batch_size=self.base_mlp.batch_size,
            callbacks=[
                skorch.callbacks.EarlyStopping(monitor='valid_loss', patience=10, lower_is_better=True)],
            verbose=True
        )
        self.model.initialize()
        self.calibrate_model(calibration_set=validation_set, **kwargs)

    def predict_proba(self, X_input):
        X_input = X_input.to(self.cm_device)
        return self.model.predict_proba(X_input)[:, 1]

    def save(self, concept_model_dir: str, overwrite: bool = False):
        # Ensure the directory exists
        save_dir = os.path.join(concept_model_dir, self.str_desc)
        assert self.model is not None
        if not overwrite and os.path.exists(save_dir):
            raise FileExistsError(f"File {save_dir} already exists. Set overwrite=True to overwrite.")
        os.makedirs(save_dir, exist_ok=True)
        # save torch parameters
        sklearn_save_path = os.path.join(save_dir, "sklearn.joblib")
        dump_joblib(self.model, sklearn_save_path)
        # save model parameters
        model_state_save_path = os.path.join(save_dir, "model_state.pkl")
        with open(model_state_save_path, 'wb') as outp:  # Overwrites any existing file.
            state = {k: v for (k, v) in self.__dict__.items() if not k.startswith('_')}
            state.pop('base_mlp', None)
            dill.dump(state, outp, dill.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, load_dir: str):
        # todo put in super class
        if not os.path.exists(load_dir):
            raise FileNotFoundError(f"Directory {load_dir} does not exist.")
        model_instance = cls()  # You might need to supply initial parameters to the constructor
        # Load model parameters
        model_state_load_path = os.path.join(load_dir, "model_state.pkl")
        with open(model_state_load_path, 'rb') as inp:
            state = dill.load(inp)
            state.pop('base_mlp', None)
            model_instance.__setstate__(state)
        # Load sklearn model
        sklearn_load_path = os.path.join(load_dir, "sklearn.joblib")
        model_instance.model = load_joblib(sklearn_load_path)  # Load model parameters
        return model_instance


# todo move to another file
def load_concept_model(concept_model_load_path: str):
    """
    Load a concept model from a file
    :param concept_model_load_path:
    :return:
    """
    filename = os.path.basename(concept_model_load_path)
    model_type = filename.split("type")[1]
    # Initialize and load the appropriate type of model
    if model_type == 'linearsvm':
        model = LinearSVMConceptModel.load(concept_model_load_path)
    elif model_type == 'logreg':
        model = LogRegConceptModel.load(concept_model_load_path)
    elif model_type == 'mlp':
        if os.path.exists(os.path.join(concept_model_load_path, "torch_state_dict.pth")):
            model = MLPConceptModel.load(concept_model_load_path)
        else:
            model = MLPConceptModelCalibrated.load(concept_model_load_path)
    elif 'finetuned' in model_type:
        model = FinetunedConceptModel.load(concept_model_load_path)
    else:
        raise ValueError(f"Unknown model type {model_type}")
    return model


def load_concept_models(concept_models_dir: str) -> List[ConceptModel]:
    """
    Load concept models
    :param concept_models_dir: directory with concept models with file names matching
    'cm{concept_index}_{concept_name}_noise{concept_noise_prob}_{model_type}'
    :return: List[ConceptModel]
    """
    loaded_models_dict = {}

    for dir_name in os.listdir(concept_models_dir):
        # print('loading model: ', dir_name, '...')
        start_time = time.time()
        concept_model_load_path = os.path.join(concept_models_dir, dir_name)
        model = load_concept_model(concept_model_load_path)
        concept_index = model.concept_index
        if concept_index in loaded_models_dict:
            raise ValueError(f"Multiple models found for concept index {concept_index}")
        loaded_models_dict[concept_index] = model
        # print('model loaded in ', time.time() - start_time, ' seconds')

    # Check if any concept index is missing and that there are no duplicates
    for i in range(len(loaded_models_dict)):
        if i not in loaded_models_dict:
            raise ValueError(f"Missing model for concept index {i}")

    # Sort by concept_index and create a list
    sorted_indices = sorted(loaded_models_dict.keys())
    loaded_models = [loaded_models_dict[i] for i in sorted_indices]

    return loaded_models


def train_concept_models(train_file_path: str,
                         validation_file_path: str,
                         model_type: str,
                         concept_models_dir: str,
                         **kwargs):
    """
    Train concept models
    :param train_file_path:
    :param validation_file_path:
    :param model_type:
    :param concept_models_dir:
    :param kwargs:
    :return:
    """
    os.makedirs(concept_models_dir, exist_ok=True)
    retrain = kwargs.pop('retrain_models', False)
    with open(train_file_path, 'rb') as train_file:
        training_set = dill.load(train_file)
    metrics_table = PrettyTable()
    metrics_table.field_names = ["Concept",
                                 "Train Samples", "Val Samples", "Test Samples",
                                 "Train Log Loss", "Val Log Loss", "Test Log Loss",
                                 "Train Error", "Val Error", "Test Error",
                                 "Train ECE", "Val ECE", "Test ECE",
                                 "Train F1", "Val F1", "Test F1",
                                 "Train AUCROC", "Val AUCROC", "Test AUCROC",
                                 "Train Brier", "Val Brier", "Test Brier"]
    train_sums_metric_dict = defaultdict(float)
    val_sums_metric_dict = defaultdict(float)
    test_sums_metric_dict = defaultdict(float)
    n_concepts = training_set.n_concepts
    for c_i in range(n_concepts):
        if training_set.concept_names is None:
            print('concept ', c_i)
        else:
            print('concept ', training_set.concept_names[c_i])
        if model_type == 'linearsvm':
            m = LinearSVMConceptModel(concept_index=c_i, **kwargs)
        elif model_type == 'mlp':
            if kwargs.get('calibrate_mlp', False):
                m = MLPConceptModelCalibrated(concept_index=c_i, **kwargs)
            else:
                m = MLPConceptModel(concept_index=c_i, **kwargs)
        elif model_type == 'logreg':
            m = LogRegConceptModel(concept_index=c_i, **kwargs)
        elif model_type == 'finetuned':
            m = FinetunedConceptModel(concept_index=c_i, **kwargs)
        else:
            raise ValueError(f"Unknown model type {model_type}")
        model_path = os.path.join(concept_models_dir, m.str_desc)
        if retrain or not os.path.exists(os.path.join(model_path, 'model_state.pkl')):
            m.save_path = model_path
            m.fit_from_paths(train_file_path, validation_file_path, save_path=model_path, **kwargs)
            m.save(concept_models_dir, overwrite=True)
        m = load_concept_model(model_path)
        m.save_path = model_path
        if kwargs.get('print_stats', False):
            use_cached_predictions = kwargs.get('use_cached_predictions', False)
            if retrain:
                use_cached_predictions = False
            train_metrics_dict = m.evaluate_dataset_from_path(train_file_path,
                                                              use_cached_predictions=use_cached_predictions)
            print(f'train_set_size: {get_metric_str(train_metrics_dict, "n_samples", digits=0)}',
                  f'train_log_loss: {get_metric_str(train_metrics_dict, "log_loss")} '
                  f'train_err: {get_metric_str(train_metrics_dict, "error")} '
                  f'train_ece: {get_metric_str(train_metrics_dict, "ece")}',
                  f'train_f1: {get_metric_str(train_metrics_dict, "f1")}',
                  f'train_auc_roc: {get_metric_str(train_metrics_dict, "auc_roc")}',
                  f'train_brier: {get_metric_str(train_metrics_dict, "brier")}',
                  )
            val_metrics_dict = m.evaluate_dataset_from_path(validation_file_path,
                                                            use_cached_predictions=use_cached_predictions)
            print(f'val_set_size: {get_metric_str(val_metrics_dict, "n_samples", digits=0)}',
                  f'val_log_loss: {get_metric_str(val_metrics_dict, "log_loss")} '
                  f'val_err: {get_metric_str(val_metrics_dict, "error")} '
                  f'val_ece: {get_metric_str(val_metrics_dict, "ece")}',
                  f'val_f1: {get_metric_str(val_metrics_dict, "f1")}',
                  f'val_auc_roc: {get_metric_str(val_metrics_dict, "auc_roc")}',
                  f'val_brier: {get_metric_str(val_metrics_dict, "brier")}',
                  )
            test_metrics_dict = None
            if kwargs.get('test_file_path', None) is not None:
                test_metrics_dict = m.evaluate_dataset_from_path(kwargs['test_file_path'],
                                                                 use_cached_predictions=use_cached_predictions)
                print(f'test_set_size: {get_metric_str(test_metrics_dict, "n_samples", digits=0)}',
                      f'test_log_loss: {get_metric_str(test_metrics_dict, "log_loss")} '
                      f'test_err: {get_metric_str(test_metrics_dict, "error")} '
                      f'test_ece: {get_metric_str(test_metrics_dict, "ece")}',
                      f'test_f1: {get_metric_str(test_metrics_dict, "f1")}',
                      f'test_auc_roc: {get_metric_str(test_metrics_dict, "auc_roc")}',
                      f'test_brier: {get_metric_str(test_metrics_dict, "brier")}',
                      )
            for metric_name in ["log_loss", "error", "ece", "f1", "auc_roc", "brier"]:
                train_sums_metric_dict[metric_name] += train_metrics_dict.get(metric_name, 0)
                val_sums_metric_dict[metric_name] += val_metrics_dict.get(metric_name, 0)
                if test_metrics_dict:
                    test_sums_metric_dict[metric_name] += test_metrics_dict.get(metric_name, 0)
            if training_set.concept_names is None:
                c_i_name = str(c_i)
            else:
                c_i_name = training_set.concept_names[c_i]
            row = [c_i_name,
                   get_metric_str(train_metrics_dict, "n_samples", digits=0),
                   get_metric_str(val_metrics_dict, "n_samples", digits=0),
                   get_metric_str(test_metrics_dict, "n_samples", digits=0),
                   get_metric_str(train_metrics_dict, "log_loss"),
                   get_metric_str(val_metrics_dict, "log_loss"),
                   get_metric_str(test_metrics_dict, "log_loss"),
                   get_metric_str(train_metrics_dict, "error"),
                   get_metric_str(val_metrics_dict, "error"),
                   get_metric_str(test_metrics_dict, "error"),
                   get_metric_str(train_metrics_dict, "ece"),
                   get_metric_str(val_metrics_dict, "ece"),
                   get_metric_str(test_metrics_dict, "ece"),
                   get_metric_str(train_metrics_dict, "f1"),
                   get_metric_str(val_metrics_dict, "f1"),
                   get_metric_str(test_metrics_dict, "f1"),
                   get_metric_str(train_metrics_dict, "auc_roc"),
                   get_metric_str(val_metrics_dict, "auc_roc"),
                   get_metric_str(test_metrics_dict, "auc_roc"),
                   get_metric_str(train_metrics_dict, "brier"),
                   get_metric_str(val_metrics_dict, "brier"),
                   get_metric_str(test_metrics_dict, "brier"),
                   ]
            metrics_table.add_row(row)
    print(f'model: {model_type}, dataset: {kwargs["dataset"]}')
    # divide train_sums_metric_dict, val_sums_metric_dict and test_sums_metric_dict by n_concepts to get averages
    avg_train_metrics = {key: val / n_concepts for key, val in train_sums_metric_dict.items()}
    avg_val_metrics = {key: val / n_concepts for key, val in val_sums_metric_dict.items()}
    avg_test_metrics = {key: val / n_concepts for key, val in test_sums_metric_dict.items() if test_metrics_dict}

    # Add averages to metrics_table
    avg_metrics_row = ['Average',
                       "-",  # Placeholder for "Train Log Loss", "Val Log Loss", "Test Log Loss"
                       "-",  # Placeholder for "Train Error", "Val Error", "Test Error"
                       "-",  # Placeholder for "Train ECE", "Val ECE", "Test ECE"
                       avg_train_metrics.get("log_loss", "-"),
                       avg_val_metrics.get("log_loss", "-"),
                       avg_test_metrics.get("log_loss", "-"),
                       avg_train_metrics.get("error", "-"),
                       avg_val_metrics.get("error", "-"),
                       avg_test_metrics.get("error", "-"),
                       avg_train_metrics.get("ece", "-"),
                       avg_val_metrics.get("ece", "-"),
                       avg_test_metrics.get("ece", "-"),
                       avg_train_metrics.get("f1", "-"),
                       avg_val_metrics.get("f1", "-"),
                       avg_test_metrics.get("f1", "-"),
                       avg_train_metrics.get("auc_roc", "-"),
                       avg_val_metrics.get("auc_roc", "-"),
                       avg_test_metrics.get("auc_roc", "-"),

                       avg_train_metrics.get("brier", "-"),
                       avg_val_metrics.get("brier", "-"),
                       avg_test_metrics.get("brier", "-")]
    metrics_table.add_row(avg_metrics_row)
    print(metrics_table)
    print(f'Saved models to {concept_models_dir}')
