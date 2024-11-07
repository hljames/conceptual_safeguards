import warnings
from typing import Optional

from numpy import ndarray
from sklearn.metrics import log_loss, f1_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, \
    cohen_kappa_score, brier_score_loss

import numpy as np
from torch import tensor
from torchmetrics.classification import BinaryCalibrationError, MulticlassCalibrationError


# ece = calibration_error(y_true, y_pred, num_bins=10)

def ece(y_true, y_pred_proba, num_bins=15, bce_norm='l1'):
    num_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 2
    if num_classes > 2:
        mce = MulticlassCalibrationError(n_bins=num_bins, norm=bce_norm, num_classes=num_classes)
        return mce(tensor(y_pred_proba), tensor(y_true)).item()
    else:
        bce = BinaryCalibrationError(n_bins=num_bins, norm=bce_norm)
        return bce(tensor(y_pred_proba), tensor(y_true)).item()


# def ece(y_true: np.ndarray, y_pred_proba: np.ndarray, strategy: str = 'uniform') -> float:
#     n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else 1
#
#     ece_all_classes = 0
#     n_samples = len(y_true)
#
#     for c in range(n_classes):
#         y_true_binary = (y_true == c).astype(int)
#         y_pred_proba_binary = y_pred_proba[:, c] if n_classes > 1 else y_pred_proba
#
#         df = pd.DataFrame({'target': y_true_binary, 'proba': y_pred_proba_binary, 'bin': np.nan})
#
#         if strategy == 'uniform':
#             lim_inf = np.linspace(0, 0.9, 10)
#             for idx, lim in enumerate(lim_inf):
#                 df.loc[df['proba'] >= lim, 'bin'] = idx
#
#         if df['bin'].isna().any():
#             print("Warning: 'bin' column has NaN values")
#
#         value_counts_df = df['bin'].value_counts().reset_index()
#         value_counts_df.columns = ['bin', 'count']
#
#         mean_df = df.groupby('bin').mean().reset_index()
#         df_bin_groups = pd.merge(mean_df, value_counts_df, on='bin')
#
#         df_bin_groups['ece'] = np.abs(df_bin_groups['target'] - df_bin_groups['proba']) * (
#                 df_bin_groups['count'] / n_samples)
#         ece_class_c = df_bin_groups['ece'].sum()
#
#         ece_all_classes += ece_class_c * (np.sum(y_true == c) / n_samples)
#
#     return ece_all_classes


def error(y_true: ndarray, y_pred: ndarray) -> float:
    """
    Calculates the error rate between two binary numpy arrays of shape n

    Parameters:
    y_true: numpy array of binary values of shape (n,)
        The true labels
    y_pred: numpy array of binary values of shape (n,)
        The predicted labels

    Returns:
    float
        The error rate between y_true and y_pred
    """
    n = len(y_true)
    er = np.sum(y_true != y_pred) / n
    return er


def get_metric_str(metrics_dict, metric_name, digits=5):
    return format(metrics_dict[metric_name], f".{digits}f") if metrics_dict and metric_name in metrics_dict else "N/A"


def calculate_metrics(y_true: ndarray,
                      y_pred: ndarray,
                      y_pred_proba: ndarray,
                      y_true_proba: Optional[ndarray] = None,
                      labels: Optional[ndarray] = None) -> dict:
    # cast all as numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    if y_true_proba is not None:
        y_true_proba = np.array(y_true_proba)
    is_binary = y_pred_proba.shape[1] == 1
    # n_classes = y_pred_proba.shape[1]
    # is_binary = n_classes == 2
    #
    # # Extract probabilities for the positive class in the binary case
    # if is_binary:
    #     y_pred_proba = y_pred_proba[:, 1]
    #     if y_true_proba is not None:
    #         y_true_proba = y_true_proba[:, 1]

    metrics = {
        'n_samples': len(y_true),
        'log_loss': np.nan,
        'error': np.nan,
        'ece': np.nan,
        'm_diff': np.nan,
        'f1': np.nan,
        'auc_roc': np.nan,
        'mcc': np.nan,
        'kappa': np.nan,
        'percent_positives': np.nan,
        'brier': np.nan,
    }

    if len(np.unique(y_true)) > 1 and len(y_true) > 1:  # At least two classes and two samples required
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics['log_loss'] = log_loss(y_true, y_pred_proba, labels=labels)
        metrics['error'] = (y_true != y_pred).mean()
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

        metrics['ece'] = ece(y_true, y_pred_proba) #strategy='uniform')

    if y_true_proba is not None and len(y_pred_proba) > 1 and len(y_true_proba) > 1:
        metrics['m_diff'] = abs(y_pred_proba - y_true_proba).mean()

    if is_binary:  # Only valid for binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['auc_pr'] = auc(recall, precision)
        metrics['percent_positives'] = y_true.mean()
        metrics['brier'] = brier_score_loss(y_true=y_true, y_prob=y_pred_proba)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted' if not is_binary else 'binary')
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo' if not is_binary else 'raise',
                                           average='weighted', labels=labels)

    return metrics
