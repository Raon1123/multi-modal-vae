import torch
<<<<<<< HEAD
import torch.nn as nn

from models.vae import MNISTVAE, CIFARVAE
from models.cvae import MNISTCVAE, CIFARCVAE
from models.vae_vanilla import MNISTVAEVanilla, CIFARVAEVanilla
=======

import models.vae as vae
import models.mmvae as mmvae
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89

def load_model(loading_path):
    model = torch.load(loading_path)
    return model


def get_model(config):
    model_configs = config['MODEL']

<<<<<<< HEAD
    model_name_dict = {'CIFAR10VAE': CIFARVAE, 
                       'CIFAR100VAE': CIFARVAE, 
                       'MNISTVAEVanilla': MNISTVAEVanilla,
                       'CIFAR10VAEVanilla': CIFARVAEVanilla, 
                       'CIFAR100VAEVanilla': CIFARVAEVanilla,
                       'MNISTVAE': MNISTVAE,
                       'MNISTCVAE': MNISTCVAE,
                       'CIFAR10CVAE': CIFARCVAE,
                       'CIFAR100CVAE': CIFARCVAE}

=======
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89
    if model_configs['pretrained']:
        try:
            model = load_model(model_configs['pretrain_path'])
        except:
            raise ValueError('Pretrained model not found.')
    else:
<<<<<<< HEAD
        model_name = model_configs['name']
        dataset_name = config['DATA']['name']
        model_class = model_name_dict[f"{dataset_name}{model_name}"]
        model = model_class(model_configs)
=======
        dataset_name = config['DATA']['name']
        if dataset_name == 'MNIST':
            model = vae.MNISTVAE(model_configs)
        elif dataset_name == 'CIFAR10':
            model = vae.CIFARVAE(model_configs)
        elif dataset_name == 'CIFAR100':
            model = vae.CIFARVAE(model_configs)
        else:
            raise NotImplementedError('Model not implemented.')

>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89
    return model


def get_optimizer(model, config):
    optim_configs = config['OPTIMIZER']
    optim_type = optim_configs['name'].lower()
    optim_params = optim_configs['params']

    if optim_type == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
    elif optim_type == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), **optim_params)
    else:
        raise ValueError('Optimizer type not recognized.')

    return optimizer

<<<<<<< HEAD
# get resnet18 classifier
def get_classifier(config):
    model_path = config['MODEL']['classifier_path']
    dataset_name = config['DATA']['name']

    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
    if dataset_name == 'CIFAR10':
        model.fc = nn.Linear(512, 10)
    elif dataset_name == 'CIFAR100':
        model.fc = nn.Linear(512, 100)
    elif dataset_name == 'MNIST':
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(512, 10)

    model.load_state_dict(torch.load(model_path))
    print(f'loaded pretrained classifier weights from {model_path}')
    return model
=======
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89
