import argparse
import yaml

import numpy as np
import torch
import tqdm

from mmdatasets.datautils import get_dataloader
from models.modelutils import get_model, get_optimizer, get_classifier
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
    model_type = 'vae' if config['MODEL']['name'] == 'VAE' else 'cvae'

    # Get data, optimizer, criteria
    train_loader, test_loader = get_dataloader(config)
    optimizer = get_optimizer(model, config)
    criteria, t_criteria = get_criteria(config)
    aux_objective = ''
    classifier = None
    if model_type == 'cave':
        aux_objective = config['MODEL']['aux_objective']
        if 'cls' in aux_objective:
            classifier = get_classifier(config).to(device)
            classifer.eval()

    # logging
    log_path = config['LOGGING']['log_path']
    save_path = config['LOGGING']['save_path']
    writer = loggings.get_writer(config)

    # Train
    pbar = tqdm.tqdm(range(config['OPTIMIZER']['epochs']))
    for epoch in pbar:
        train_loss = epochs.train_epoch(train_loader, model, optimizer, criteria, device=device, 
                                        model_type=model_type, aux_objective='entropy', classifier=None)
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