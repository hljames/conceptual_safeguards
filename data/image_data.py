import os
import random
from copy import deepcopy

import numpy as np
import torch

from numpy import ndarray
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from typing import List

from tqdm import tqdm

from data.data import CBMDataset


class ImageCBMDataset(CBMDataset):
    """
    Image CBM Dataset
    """

    def __init__(self,
                 X: List,
                 C: ndarray,
                 y: ndarray,
                 **kwargs):
        super().__init__(X, C, y, **kwargs)
        self.preprocess = kwargs.get('preprocess', None)
        self.base_dir = kwargs.get('base_dir', None)
        self.is_train = kwargs.get('is_train', False)

    def augment_image(self, img):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(np.random.uniform(0.5, 1.5))  # Random color enhancement

        if np.random.rand() > 0.5:  # Random horizontal flip
            img = ImageOps.mirror(img)

        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = np.array(self.X)[idx]
        if self.base_dir is not None:
            img_path = os.path.join(self.base_dir, img_path)
        try:
            image = Image.open(img_path).convert('RGB')
            if self.preprocess:
                image = self.preprocess(image)
        except AttributeError as e:
            print(f'WARNING: {e}')
            print('cannot open multiple images, returning paths')
            image = img_path
        C_idx = torch.from_numpy(np.array(self.C[idx, :], dtype=np.int32))
        y_idx = torch.from_numpy(np.array(self.y[idx], dtype=np.int32))
        return image, C_idx, y_idx


def augment_image(image_path, save_path):
    assert not os.path.exists(save_path), f"Save path {save_path} already exists"
    orig_img = Image.open(image_path)
    img = orig_img.copy()

    # Apply a random rotation
    angle = random.randint(-30, 30)
    img = img.rotate(angle)

    # Apply a random brightness change
    brightness_factor = random.uniform(0.7, 1.3)  # between 70% and 130%
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # Apply a random contrast change
    contrast_factor = random.uniform(0.7, 1.3)  # between 70% and 130%
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Optionally, apply horizontal flip with 50% probability
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Optionally, apply a slight Gaussian blur
    if random.random() > 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # Optionally, apply zoom (crop and resize)
    if random.random() > 0.5:
        x1 = int(img.width * 0.1)
        y1 = int(img.height * 0.1)
        x2 = img.width - x1
        y2 = img.height - y1
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((img.width, img.height), Image.BILINEAR)
    img.save(save_path)


def augment_training_dataset(train_dataset, augmentation_factor):
    augmented_dataset = deepcopy(train_dataset)
    if augmentation_factor <= 1.0:
        return augmented_dataset
    total_images = len(augmented_dataset.X)
    total_augmented_images = int(total_images * augmentation_factor) - total_images

    augmented_X, augmented_C, augmented_y = [], [], []
    assert augmented_dataset.base_dir is not None
    base_dir = augmented_dataset.base_dir
    aug_imgs_dir = os.path.join(base_dir, 'augmented_images')
    os.makedirs(aug_imgs_dir, exist_ok=True)
    for i in tqdm(range(total_augmented_images)):
        # Randomly pick an original image to augment
        idx = random.randint(0, total_images - 1)

        original_filename = augmented_dataset.X[idx]
        aug_fname = os.path.join('augmented_images', f"aug_{i}_{os.path.basename(original_filename)}")
        # Create a new filename for the augmented image
        original_image_path = os.path.join(base_dir, original_filename)
        augmented_image_path = os.path.join(base_dir, aug_fname)
        if not os.path.exists(augmented_image_path):
            os.makedirs(os.path.dirname(augmented_image_path), exist_ok=True)
            # Perform the augmentation
            augment_image(original_image_path, augmented_image_path)

        # Add the augmented data to the dataset
        augmented_X.append(aug_fname)
        augmented_C.append(augmented_dataset.C[idx])
        augmented_y.append(augmented_dataset.y[idx])

    # Convert lists to NumPy arrays for concatenation
    augmented_X = np.array(augmented_X)
    augmented_C = np.array(augmented_C)
    augmented_y = np.array(augmented_y)

    # Merge original and augmented data
    augmented_dataset.X = np.concatenate((augmented_dataset.X, augmented_X))
    augmented_dataset.C = np.concatenate((augmented_dataset.C, augmented_C))
    augmented_dataset.y = np.concatenate((augmented_dataset.y, augmented_y))

    return augmented_dataset
