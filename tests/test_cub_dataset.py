import dill
import numpy as np
from sklearn.linear_model import LogisticRegression

from metrics import calculate_metrics

multiclass_train_file_path = '/Users/hailey/repos/uacbm/datasets/cub/training.pkl'
multiclass_validation_file_path = '/Users/hailey/repos/uacbm/datasets/cub/validation.pkl'
multiclass_concept_models_dir = '/Users/hailey/repos/uacbm/saved_models/cub/concept_models'

with open(multiclass_train_file_path, 'rb') as f:
    train_dataset = dill.load(f)

multiclass = len(np.unique(train_dataset.y)) > 2

y_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
y_model.fit(train_dataset.C, train_dataset.y)
y_pred_proba = y_model.predict_proba(train_dataset.C)
y_pred = y_model.predict(train_dataset.C)

calculate_metrics(y_pred_proba=y_pred_proba, y_pred=y_pred, y_true=train_dataset.y)

