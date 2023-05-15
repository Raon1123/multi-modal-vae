import torch

import models.vae as vae
import models.mmvae as mmvae

def load_model(loading_path):
    model = torch.load(loading_path)
    return model


def get_model(config):
    model_configs = config['MODEL']

    if model_configs['pretrained']:
        try:
            model = load_model(model_configs['pretrain_path'])
        except:
            raise ValueError('Pretrained model not found.')
    else:
        dataset_name = config['DATA']['name']
        if dataset_name == 'MNIST':
            model = vae.MNISTVAE(model_configs)
        elif dataset_name == 'CIFAR10':
            model = vae.CIFARVAE(model_configs, 10)
        elif dataset_name == 'CIFAR100':
            model = vae.CIFARVAE(model_configs, 100)
        else:
            raise NotImplementedError('Model not implemented.')

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

