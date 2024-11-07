## Most of these codes are taken from https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP and the DDI paper.
import dill
import numpy as np
import torch
import torch.nn as nn
import os
import torchvision

# google drive paths to models
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights
from tqdm import tqdm

from constants import EMBEDDING_MODEL_DIR, PROCESSED_DATASETS_DIR

MODEL_WEB_PATHS = {
    'HAM10000_INCEPTION': 'https://drive.google.com/uc?id=1ToT8ifJ5lcWh8Ix19ifWlMcMz9UZXcmo',
    'HAM10000_INCEPTION_DERM7PT_BINARY': '',
    'HAM10000_INCEPTION_DERM7PT_BINARY_AUTOENCODER': '',
    'HAM10000_INCEPTION_DERM7PT': '',
    'HAM10000_INCEPTION_DERM7PT_AUTOENCODER': '',
}

# thresholds determined by maximizing F1-score on the test split of the train 
#   dataset for the given algorithm
MODEL_THRESHOLDS = {
    'HAM10000_INCEPTION': 0.733,
    'HAM10000_INCEPTION_DERM7PT_BINARY': 0.733,
    'HAM10000_INCEPTION_DERM7PT_BINARY_AUTOENCODER': 0.733,
    'HAM10000_INCEPTION_DERM7PT': 0.733,
    'HAM10000_INCEPTION_DERM7PT_AUTOENCODER': 0.733,
}


def fine_tune_model(model, dataset_dir, epochs=20, lr=0.001, batch_size=32, num_workers=None, device='mps',
                    as_autoencoder=False):
    """
    Fine-tune the Inception model on the given dataset.
    :param model:
    :param dataset_dir:
    :param epochs:
    :param lr:
    :param batch_size:
    :param num_workers:
    :return:
    """
    if num_workers is None:
        num_workers = os.cpu_count() - 1
    train_file_path = os.path.join(dataset_dir, 'training.pkl')
    with open(train_file_path, 'rb') as train_file:
        training_dataset = dill.load(train_file)
    training_dataset.preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if as_autoencoder:
        raise NotImplementedError
        # criterion = nn.MSELoss()
        # model.fc = nn.Linear(2048, 2048)
        # model.AuxLogits.fc = nn.Linear(768, 768)
    else:
        criterion = nn.CrossEntropyLoss()
        num_classes = len(np.unique(training_dataset.y))
        model.fc = torch.nn.Linear(2048, num_classes)
        model.AuxLogits.fc = torch.nn.Linear(768, num_classes)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        # correct = 0
        # total = 0
        for i, data in tqdm(enumerate(train_dataloader, 0)):
            inputs, _, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device) if not as_autoencoder else inputs.to(device)
            optimizer.zero_grad()
            outputs, aux_outputs = model(inputs)
            if as_autoencoder:
                loss = criterion(outputs, labels)
            else:
                loss1 = criterion(outputs, labels.long())
                loss2 = criterion(aux_outputs, labels.long())
                loss = loss1 + 0.4 * loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(train_dataloader)
        # accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")  # , Accuracy: {accuracy:.2f}%")
    print("Finished fine-tuning")
    return model


def load_model(backbone_name, download=True, device='mps'):
    # Taken from the DDI repo https://drive.google.com/drive/folders/1oQ53WH_Tp6rcLZjRp_-UBOQcMl-b1kkP
    """Load the model and download if necessary. Saves model to provided save 
    directory."""
    model_path = os.path.join(EMBEDDING_MODEL_DIR, f"{backbone_name.lower()}.pth")
    ham10000_inception_model_path = os.path.join(EMBEDDING_MODEL_DIR, "ham10000_inception.pth")
    if backbone_name.lower() == 'inception':
        model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=True)
        model.fc = torch.nn.Linear(2048, 2)
        model.AuxLogits.fc = torch.nn.Linear(768, 2)
        return model
    else:

        # download ham10000_inception model if necessary
        if not os.path.exists(ham10000_inception_model_path):
            if not download:
                raise Exception("Model not downloaded and download option not" \
                                " enabled.")
            else:
                # Requires installation of gdown (pip install gdown)
                import gdown
                gdown.download(MODEL_WEB_PATHS[backbone_name], ham10000_inception_model_path)
        model = torchvision.models.inception_v3(init_weights=False, weights=None, transform_input=True)
        model.fc = torch.nn.Linear(2048, 2)
        model.AuxLogits.fc = torch.nn.Linear(768, 2)
        model.load_state_dict(torch.load(ham10000_inception_model_path))
        # load ham10000_inception or ham10000_inception fine-tuned on derm7pt
        if backbone_name.lower() == 'ham10000_inception' or (
                'ham10000_inception_derm7pt' in backbone_name.lower() and not os.path.exists(model_path)):
            model._ddi_name = backbone_name
            model._ddi_threshold = MODEL_THRESHOLDS[backbone_name]
            model._ddi_web_path = MODEL_WEB_PATHS[backbone_name]
            if 'ham10000_inception_derm7pt' in backbone_name.lower():
                # finetune the ham10000_inception model on derm7pt
                use_binary = 'binary' in backbone_name.lower()
                fine_tune_as_autoencoder = 'autoencoder' in backbone_name.lower()
                dataset_path = os.path.join(PROCESSED_DATASETS_DIR, 'derm7pt')
                if use_binary:
                    dataset_path = os.path.join(dataset_path, 'binary')
                model = fine_tune_model(model, dataset_path, device=device, as_autoencoder=fine_tune_as_autoencoder)
                # save model
                torch.save(model.state_dict(), model_path)
        return model


class InceptionBottom(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionBottom, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        self.layer = layer
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.features = nn.Sequential(*all_children[:until_layer])
        self.model = original_model

    def _transform_input(self, x):
        x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 model.x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 model.x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 model.x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 model.x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 model.x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192model. x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192model. x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256model. x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288model. x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768model. x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768model. x 17 x 17
        # N x 768model. x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 128model.0 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 204model.8 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 204model.8 x 8 x 8
        # Adaptivmodel.e average pooling
        x = self.model.avgpool(x)
        # N x 204model.8 x 1 x 1
        x = self.model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        return x


class InceptionTop(nn.Module):
    def __init__(self, original_model, layer="penultimate"):
        super(InceptionTop, self).__init__()
        layer_dict = {"penultimate": -2,
                      "block_6": -4,
                      "block_5": -5,
                      "block_4": -6}
        until_layer = layer_dict[layer]
        all_children = list(original_model.children())
        all_children.insert(-1, nn.Flatten(1))
        self.layer = layer
        self.features = nn.Sequential(*all_children[until_layer:])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x


def get_derma_model(backbone_name="ham10000", device="mps"):
    model = load_model(backbone_name.upper(), device=device)
    model = model.to(device)
    model = model.eval()
    model_bottom, model_top = InceptionBottom(model), InceptionTop(model)
    return model, model_bottom, model_top
