import dill
import numpy as np
from sklearn.linear_model import LogisticRegression

from models.concept_bottleneck_model import PropagationCBM, IndependentCBM

# binary_train_file_path = '/Users/hailey/repos/uacbm/datasets/cub/10_concepts/binary/training.pkl'
# binary_validation_file_path = '/Users/hailey/repos/uacbm/datasets/cub/10_concepts/binary/validation.pkl'
# binary_concept_models_dir = '/Users/hailey/repos/uacbm/saved_models/cub/10_concepts/binary/concept_models'
#
# binary_icbm = IndependentCBM(concept_models_dir=binary_concept_models_dir, multiclass=False)
# binary_icbm.fit_from_paths(binary_train_file_path, binary_validation_file_path)
#
# binary_uacbm = PropagationCBM(concept_models_dir=binary_concept_models_dir, multiclass=False)
# binary_uacbm.fit_from_paths(binary_train_file_path, binary_validation_file_path)
#
# metrics = binary_icbm.evaluate_dataset_from_path(binary_train_file_path)
# print('INDEPENDENT CBM', metrics)
#
# metrics = binary_uacbm.evaluate_dataset_from_path(binary_train_file_path)
# print('UACBM', metrics)

multiclass_train_file_path = '/Users/hailey/repos/uacbm/datasets/cub/training.pkl'
multiclass_validation_file_path = '/Users/hailey/repos/uacbm/datasets/cub/validation.pkl'
multiclass_concept_models_dir = '/Users/hailey/repos/uacbm/saved_models/cub/concept_models'


with open(multiclass_train_file_path, 'rb') as f:
    train_dataset = dill.load(f)

multiclass_uacbm = PropagationCBM(concept_models_dir=multiclass_concept_models_dir, multiclass=True)
multiclass_uacbm.fit_from_paths(multiclass_train_file_path, multiclass_validation_file_path)

metrics = multiclass_uacbm.evaluate_dataset_from_path(multiclass_validation_file_path)
print('UACBM', metrics)

multiclass_icbm = IndependentCBM(concept_models_dir=multiclass_concept_models_dir, multiclass=True)
multiclass_icbm.fit_from_paths(multiclass_train_file_path, multiclass_validation_file_path)

metrics = multiclass_icbm.evaluate_dataset_from_path(multiclass_validation_file_path)
print('INDEPENDENT CBM', metrics)


