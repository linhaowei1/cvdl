import os
import pdb
import numpy as np
import torch
import pdb
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split
from utils.utils import set_random_seed

DATA_PATH = '/home/linhw/myproject/cvdl_data'

def get_transform():
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    return data_transform['train'], data_transform['test']

def get_dataset(P, test_only=False):
    
    train_transform, test_transform = get_transform()
    
    if test_only:
        test_dir = os.path.join(DATA_PATH, 'test')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    else:
        train_dir = os.path.join(DATA_PATH, 'train')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        val_set = datasets.ImageFolder(train_dir, transform=test_transform)
        train_ind, val_ind = get_indices_with_len(train_set, P.val_size)
        train_set = Subset(train_set, train_ind)
        val_set = Subset(test_set, val_ind)

def get_indices_with_len(dataset, length=10):
    set_random_seed(0)
    dataset_size = len(dataset)
    val_ind = []
    train_ind = []
    classes = [0] * 100
    for idx, data in enumerate(dataset):
        if classes[data[1]] < length:
            classes[data[1]] += 1
            val_ind.append(idx)
        else:
            train_ind.append(idx)
    return train_ind, val_ind
