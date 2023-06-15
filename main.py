import argparse
import os
import yaml

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import tqdm

from mmdatasets.datautils import get_dataloader
from models.modelutils import get_model, get_optimizer, get_classifier
from utils.criteria import get_criteria

import utils.epochs as epochs
import utils.logging as logging

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
    model_type = config['MODEL']['name'].lower()

    # Get data, optimizer, criteria
    train_loader, test_loader = get_dataloader(config)
    optimizer = get_optimizer(model, config)
    
    if 'schedules' in config['OPTIMIZER']:
        scheduler = MultiStepLR(optimizer, milestones=config['OPTIMIZER']['schedules'], gamma=0.1)
    else:
        scheduler = None

    criteria, t_criteria = get_criteria(config)
    aux_objective = ''
    classifier = None
    if model_type == 'cvae':
        aux_objective = config['MODEL']['aux_objective']
        classifier = get_classifier(config).to(device)
        classifier.eval()

    # logging
    log_path = config['LOGGING']['log_path']
    save_path = config['LOGGING']['save_path']

    exp_name = logging.exp_str(config)
    save_path = os.path.join(save_path, exp_name)

    # Create directories
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    writer = logging.get_writer(config)

    # Train
    pbar = tqdm.tqdm(range(config['OPTIMIZER']['epochs']))
    for epoch in pbar:
        train_loss = epochs.train_epoch(train_loader, model, optimizer, criteria, device=device, 
                                        model_type=model_type, aux_objective=aux_objective, classifier=classifier)
        if scheduler is not None:
            scheduler.step()
        test_loss = epochs.test_epoch(test_loader, model, t_criteria, device=device)
        
        logging.log_recon_analysis(model, test_loader, save_path, epoch, device=device)
        logging.log_scalars(writer, train_loss, test_loss, epoch)

        pbar.set_description(f'Epoch {epoch+1}: train loss {train_loss:.4f}, test loss {test_loss:.4f}')

        model.generate(run_path=save_path, epoch=epoch)

    # Save model
    logging.save_model(model, config)

    # Generate image
    model.generate(run_path=save_path, epoch=epoch)

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    print(config)
    main(config)