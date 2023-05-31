import argparse
import yaml

import numpy as np
import torch
import tqdm

from mmdatasets.datautils import get_dataloader
from models.modelutils import get_model, get_optimizer
from utils.criteria import get_criteria

import utils.epochs as epochs
import utils.logging as loggings

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-modal VAE')
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='Path to the config file.')
    
    return parser.parse_args()


def main(config):
    # Set random seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config)
    model.to(device)

    # Get data, optimizer, criteria
    train_loader, test_loader = get_dataloader(config)
    optimizer = get_optimizer(model, config)
    criteria, t_criteria = get_criteria(config)

    # logging
    log_path = config['LOGGING']['log_path']
    save_path = config['LOGGING']['save_path']
    writer = loggings.get_writer(config)

    # Train
    pbar = tqdm.tqdm(range(config['OPTIMIZER']['epochs']))
    for epoch in pbar:
        train_loss = epochs.train_epoch(train_loader, model, optimizer, criteria, device=device)
        test_loss = epochs.test_epoch(test_loader, model, t_criteria, device=device)
        
        loggings.log_recon_analysis(model, test_loader, log_path, epoch, device=device)
        loggings.log_scalars(writer, train_loss, test_loss, epoch)

        pbar.set_description(f'Epoch {epoch}: train loss {train_loss:.4f}, test loss {test_loss:.4f}')

    # Save model
    loggings.save_model(model, config)
    model.generate(save_path, epoch)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    print(config)
    main(config)