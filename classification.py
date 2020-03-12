# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from models import CNN_basic
from pytorch_utils.nn_utils import train
from pytorch_utils.nn_utils import evaluate
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def read_files(str_train_f, str_test_f):
    train_data = pd.read_csv(str_train_f)
    test_data = pd.read_csv(str_test_f)
    return train_data, test_data


def init_dataloaders(training_set, test_set, **kwargs):
    train_loader = DataLoader(training_set, **kwargs)
    test_loader = DataLoader(test_set, **kwargs)
    return train_loader, test_loader


def get_target_from_labels(labels):
    targets = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        targets[i][labels[i]] = 1.
    return targets


if __name__ == '__main__':
    # Load CSV MNIST files
    training_set, test_set = read_files('train.csv', 'test.csv')
    training_set = training_set.to_numpy()
    training_labels = training_set[:, 0]
    training_target = get_target_from_labels(training_labels)
    training_set = training_set[:, 1:]
    test_set = test_set.to_numpy()
    # Reshape to torch format
    training_set = training_set.reshape((-1, 1, 28, 28))
    training_set = training_set.astype(float)
    training_set = training_set / 255
    test_set = test_set.reshape((-1, 1, 28, 28))
    training_set = TensorDataset(torch.Tensor(training_set),
                                 torch.Tensor(training_target))
    test_set = torch.Tensor(test_set)
    test_set = test_set / 255
    train_loader, test_loader = init_dataloaders(training_set, test_set,
                                                 batch_size=16)
    model = CNN_basic(lr=0.001, in_channels=1, out_channels=10,
                      optimizer=torch.optim.Adam)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    epochs = 100
    train(model, epochs, model.model.get_optim(), train_loader, device)
    out_tensor = evaluate(model, test_loader, device)
    labels = torch.argmax(out_tensor, axis=1)
    label_series = pd.Series(labels.cpu().numpy())
    label_series.reset_index(inplace=True, drop=True)
    imageid_series = pd.Series(range(1, len(labels)+1))
    imageid_series.reset_index(inplace=True, drop=True)
    sub_df = pd.DataFrame({'ImageId': imageid_series, 'Label': label_series})
    sub_df.to_csv(
        'test_submission.csv',
        index=False
    )
