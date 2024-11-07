import os.path
from typing import Optional

import dill
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm

from constants import EMEDDED_DATASETS_DIR, PROCESSED_DATASETS_DIR
from data.data import CBMDataset, downsample_to_balance_concept, downsample_by_max_samples
from models.derma_models import InceptionBottom, InceptionTop


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_model(backbone_name="resnet18_cub", out_dir='./saved_models/embedding_models', full_model=False ,
              device="mps"):
    if backbone_name == "resnet18_cub":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
        ])
    if backbone_name == "resnet18":
        from pytorchcv.model_provider import get_model as ptcv_get_model
        model = ptcv_get_model(backbone_name, pretrained=True, root=out_dir)
        backbone, model_top = ResNetBottom(model), ResNetTop(model)
        cub_mean_pxs = np.array([0.5, 0.5, 0.5])
        cub_std_pxs = np.array([2., 2., 2.])
        preprocess = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(cub_mean_pxs, cub_std_pxs)
        ])
    elif backbone_name == "inception":
        model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True)
        model.fc = torch.nn.Linear(2048, 2)
        model.AuxLogits.fc = torch.nn.Linear(768, 2)
        model.to(device)
        model.eval()
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        backbone = InceptionBottom(model)
        # model_top = InceptionTop(model)

    elif backbone_name == "resnet18_oai":
        raise NotImplementedError

    elif "ham10000_inception" in backbone_name.lower():
        from models.derma_models import get_derma_model
        model, backbone, model_top = get_derma_model(backbone_name.lower(), device=device)
        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(backbone_name)

    if full_model:
        return model, backbone, preprocess
    else:
        return backbone, preprocess


@torch.no_grad()
def get_embeddings(loader, model, device="cuda"):
    """
    Args:
        loader ([torch.utils.data.DataLoader]): Data loader returning images and labels
        model ([nn.Module]): Backbone
        device (str, optional): Device to use. Defaults to "cuda".
    Returns:
        np.array: Activations as a numpy array.
    """
    model = model.to(device)
    activations = C = y = None
    for image, C_batch, y_batch in tqdm(loader):
        image = image.to(device)
        batch_act = model(image).squeeze().detach().cpu().numpy()
        if len(batch_act.shape) == 1:
            batch_act = np.expand_dims(batch_act, axis=0)
        if activations is None:
            activations, C, y = batch_act, C_batch, y_batch
        else:
            try:
                activations = np.concatenate([activations, batch_act], axis=0)
                C = np.concatenate([C, C_batch], axis=0)
                y = np.concatenate([y, y_batch], axis=0)
            except:
                assert 0
    return activations, C, y


def embed_concept_dataset(dataset: CBMDataset,
                          embedding_backbone_name: Optional[str] = None,
                          balance_concept_index: Optional[int] = None,
                          max_samples: Optional[int] = None,
                          batch_size: int = 64,
                          num_workers: Optional[int] = None,
                          device="cuda"):
    """
    Embeds a concept dataset using a pretrained embedding model.
    :param dataset:
    :param embedding_backbone_name:
    :param balance_concept_index:
    :param max_samples:
    :param batch_size:
    :param num_workers:
    :param device:
    :return:
    """
    if num_workers is None:
        num_workers = os.cpu_count() - 1
    if balance_concept_index:
        dataset = downsample_to_balance_concept(dataset, balance_concept_index, max_samples)
    elif max_samples:
        dataset = downsample_by_max_samples(dataset, max_samples)
    if not embedding_backbone_name:
        return dataset
    embedding_backbone, preprocess = get_model(embedding_backbone_name, full_model=False, device=device)
    # update data preprocessing based on embedding model
    dataset.preprocess = preprocess
    print(f'embedding dataset with {num_workers} workers on device {device}')
    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, persistent_workers=True, pin_memory=True)
    X_embed, C, y = get_embeddings(dataset_loader, embedding_backbone, device=device)
    embedded_dataset = CBMDataset(X_embed, C, y)
    return embedded_dataset


def embed_concept_dataset_path(dataset_path: str,
                               embedding_backbone_name: Optional[str] = None,
                               balance_concept_index: Optional[int] = None,
                               max_samples: Optional[int] = None,
                               batch_size: int = 64,
                               num_workers: Optional[int] = None,
                               device="cuda",
                               verbose=False):
    """
    Load or embed a concept dataset from path using a pretrained embedding model.
    :param dataset_path:
    :param embedding_backbone_name:
    :param balance_concept_index:
    :param max_samples:
    :param batch_size:
    :param num_workers:
    :param device:
    :param verbose:
    :return:
    """
    with open(dataset_path, 'rb') as dataset_file:
        dataset = dill.load(dataset_file)
    if not embedding_backbone_name:
        return dataset
    dataset_basename = os.path.splitext(os.path.basename(dataset_path))[0]
    embedded_dataset_filename = f"{dataset_basename}_{embedding_backbone_name}"
    if balance_concept_index is not None:
        embedded_dataset_filename += f"_bi{balance_concept_index}"
    if max_samples:
        embedded_dataset_filename += f"_ms{max_samples}"
    embedded_dataset_dir = os.path.dirname(dataset_path).replace(PROCESSED_DATASETS_DIR, EMEDDED_DATASETS_DIR)
    embedded_dataset_path = os.path.join(embedded_dataset_dir, embedded_dataset_filename)
    os.makedirs(embedded_dataset_dir, exist_ok=True)
    # load and return if already embedded
    if os.path.exists(embedded_dataset_path):
        with open(embedded_dataset_path, 'rb') as embedded_dataset_file:
            embedded_dataset = dill.load(embedded_dataset_file)
            return embedded_dataset
    embedded_dataset = embed_concept_dataset(dataset, embedding_backbone_name, balance_concept_index,
                                             max_samples, batch_size, num_workers, device)
    dill.dump(embedded_dataset, open(embedded_dataset_path, 'wb'))
    print('Saved embedded dataset to: ', embedded_dataset_path)
    return embedded_dataset
