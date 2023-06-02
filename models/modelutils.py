import torch

from models.vae import MNISTVAE, CIFARVAE
from models.cvae import MNISTCVAE, CIFARCVAE

def load_model(loading_path):
    model = torch.load(loading_path)
    return model


def get_model(config):
    model_configs = config['MODEL']

    model_name_dict = {'CIFAR10VAE': CIFARVAE, 
                       'CIFAR100VAE': CIFARVAE, 
                       'MNISTVAE': MNISTVAE,
                       'MNISTCVAE': MNISTCVAE,
                       'CIFAR10CVAE': CIFARCVAE,
                       'CIFAR100CVAE': CIFARCVAE}

    if model_configs['pretrained']:
        try:
            model = load_model(model_configs['pretrain_path'])
        except:
            raise ValueError('Pretrained model not found.')
    else:
        model_name = model_configs['name']
        dataset_name = config['DATA']['name']
        model_class = model_name_dict[f"{dataset_name}{model_name}"]
        model = model_class(model_configs)
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

def get_classifier(config):
    model_path = config['MODEL']['classifier_path']

    model = torch.load(model_path)
    return model