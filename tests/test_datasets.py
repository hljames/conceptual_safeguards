import os
import dill
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append('/Users/hailey/repos/uacbm/')
from models.concept_model import load_concept_models

concept_models_dir = '/Users/hailey/repos/uacbm/saved_models/cub/concept_models_train_val_ablation/'
data_dir = '/Users/hailey/repos/uacbm/datasets/cub/'
train_file_path = os.path.join(data_dir, 'training.pkl')
validation_file_path = os.path.join(data_dir, 'validation.pkl')
test_file_path = os.path.join(data_dir, 'test.pkl')

with open(train_file_path, 'rb') as dataset_file:
    train_dataset = dill.load(dataset_file)

import torch
idx = torch.tensor([1,2,3])
X,C,y = train_dataset[idx]