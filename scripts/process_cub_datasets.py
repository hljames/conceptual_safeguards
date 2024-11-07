"""
Process CUB datasets
"""
import argparse
import dill
import numpy as np
import os
import pickle

from constants import CUB_DATA_DIR, CUB_PROCESSED_DIR, PROCESSED_DATASETS_DIR
from data.image_data import ImageCBMDataset


def config():
    default_config = {
        'random_seed': 1,
        'reduce_to_binary': 'False',
        'n_concepts': -1,
        'suffix': 'v2',
    }
    return default_config


def process_cub_datasets(metadata, save_path, n_concepts=-1, is_train=False):
    X, C, y, img_ids = [], [], [], []
    base_dir = CUB_DATA_DIR
    for im_data in metadata:
        img_path = im_data["img_path"].split('CUB_200_2011/')[1]
        # full_path = os.path.join(CUB_DATA_DIR, img_path)
        X.append(img_path)
        assert os.path.exists(os.path.join(base_dir, img_path)), f"Image path {img_path} does not exist"
        if n_concepts > 0:
            C.append(im_data["attribute_label"][:n_concepts])
        else:
            C.append(im_data["attribute_label"])
        # if reduce_to_binary:
        #     y.append(int(im_data["class_label"]>=100))
        # else:
        y.append(im_data["class_label"])
        img_ids.append(im_data["id"])
    dataset = ImageCBMDataset(X=X,
                              C=np.array(C),
                              y=np.array(y),
                              img_ids=np.array(img_ids),
                              base_dir=base_dir,
                              is_train=is_train)
    with open(save_path, 'wb') as set_file:
        dill.dump(dataset, set_file, protocol=dill.HIGHEST_PROTOCOL)
    print('Saved dataset to {}'.format(save_path))
    return dataset


if __name__ == '__main__':
    config = config()
    parser = argparse.ArgumentParser(description='Process CUB datasets')
    parser.add_argument('--random-seed', type=int, default=config['random_seed'],
                        help=f'Random seed (default {config["random_seed"]})'),
    parser.add_argument('--n-concepts', type=int, default=config['n_concepts'],
                        help=f'Number of concepts to include (default '
                             f'{"all" if config["n_concepts"] == -1 else config["n_concepts"]})')
    parser.add_argument('--suffix', type=str, default=config['suffix'],
                        help=f'Suffix to add to file name (default {config["suffix"]})')

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    assert os.path.exists(CUB_PROCESSED_DIR), f"CUB processed dir {CUB_PROCESSED_DIR} does not exist"
    train_file = os.path.join(CUB_PROCESSED_DIR, "train.pkl")
    val_file = os.path.join(CUB_PROCESSED_DIR, "val.pkl")
    train_meta_data = pickle.load(open(train_file, "rb"))
    val_meta_data = pickle.load(open(val_file, "rb"))

    # randomly split validation in half to get validation and test
    val_idx = np.random.choice(len(val_meta_data), int(len(val_meta_data) / 2), replace=False)
    test_idx = np.setdiff1d(np.arange(len(val_meta_data)), val_idx)
    test_meta_data = [val_meta_data[i] for i in test_idx]
    val_meta_data = [val_meta_data[i] for i in val_idx]

    save_dir = os.path.join(PROCESSED_DATASETS_DIR, 'cub', 'cub')
    if args.suffix:
        save_dir += f'_{args.suffix}'
    if args.n_concepts > 0:
        save_dir = os.path.join(save_dir, f'{args.n_concepts}_concepts')

    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to {save_dir}")
    # create datasets
    train_set = process_cub_datasets(train_meta_data,
                                     save_path=os.path.join(save_dir, 'training.pkl'),
                                     n_concepts=args.n_concepts, is_train=True)
    val_set = process_cub_datasets(val_meta_data,
                                   save_path=os.path.join(save_dir, 'validation.pkl'),
                                   n_concepts=args.n_concepts, is_train=False)
    test_set = process_cub_datasets(test_meta_data,
                                    save_path=os.path.join(save_dir, 'test.pkl'),
                                    n_concepts=args.n_concepts, is_train=False)
