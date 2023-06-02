import os

import torch

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def get_dataset(configs):
    data_config = configs['DATA']
    dataset_name = data_config['name']
    data_path = data_config['data_path']

    train_transform, test_transform = build_transform(data_config)

    if dataset_name == 'MNIST':
        train_dataset, test_dataset = get_mnist(data_path, train_transform, test_transform)
    elif dataset_name == 'CIFAR10':
        train_dataset, test_dataset = get_cifar(data_path, 10, train_transform, test_transform)
    elif dataset_name == 'CIFAR100':
        train_dataset, test_dataset = get_cifar(data_path, 100, train_transform, test_transform)
    else:
        raise NotImplementedError('Dataset not implemented.')
    
    return train_dataset, test_dataset


def get_dataloader(configs):
    data_config = configs['DATA']

    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    pin_memory = data_config['pin_memory']

    train_dataset, test_dataset = get_dataset(configs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, test_loader

def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

def get_mnist(data_path, train_transform, test_transform):
    train_dataset = datasets.MNIST(data_path, train=True, download=True,
                                        transform=train_transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True,
                                        transform=test_transform)
    return train_dataset, test_dataset

def get_cifar(data_path, n_class, train_transform, test_transform):
    if n_class == 10:
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True,
                                        transform=train_transform)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True,
                                            transform=test_transform)
    else:
        train_dataset = datasets.CIFAR100(data_path, train=True, download=True,
                                        transform=train_transform)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True,
                                            transform=test_transform)
    return train_dataset, test_dataset

def build_transform(configs):
    dataset_name = configs['name']
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    train_transform = []
    train_transform.append(transforms.ToTensor())
    """
    if dataset_name == 'MNIST':
        train_transform.append(transforms.Normalize(0.5, 0.5))
    else:
        train_transform.append(transforms.Normalize(mean, std))
    """
    train_transform = transforms.Compose(train_transform)

    test_transform = []
    test_transform.append(transforms.ToTensor())
    """
    if dataset_name == 'MNIST':
        test_transform.append(transforms.Normalize(0.5, 0.5))
    else:
        test_transform.append(transforms.Normalize(mean, std))
    """
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform