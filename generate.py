import argparse
import os
import yaml

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
import tqdm
from torchvision.utils import make_grid

from mmdatasets.datautils import get_dataloader
from models.modelutils import get_model, get_optimizer, get_classifier
from utils.criteria import get_criteria
from models.cvae import one_hot

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
    
    parser.add_argument('--model_path', type=str, default='', help='Trained model weight path')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate.')
    return parser.parse_args()

def generate_data(model, z, ci, model_name='VAE'):
    if model_name == 'CVAE':
        c_idxs = torch.zeros(z.size(0), dtype=torch.long, device=z.device) + ci
        c = one_hot(c_idxs, c_dim=model.c_dim).unsqueeze(1)
        x_hat = model.decoder(z, c)
    else:
        x_hat = model.decoder(z)
    x_hat = x_hat.view(-1, *x_hat.shape[-3:]).permute(0,2,3,1).cpu()
    y = torch.zeros(x_hat.size(0), dtype=torch.long) + ci
    return x_hat, y


def main(config, args):
    # Set random seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(config)

    # load model weight
    model.load_state_dict(torch.load(args.model_path))

    model.to(device)
    model.eval()
    model_type = config['MODEL']['name'].lower()

    # Get data, optimizer, criteria
    train_loader, test_loader = get_dataloader(config)

    # logging
    log_path = config['LOGGING']['log_path']
    save_path = config['LOGGING']['save_path']
    exp_name = logging.exp_str(config)
    save_path = os.path.join(save_path, exp_name)

    # Create directories
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    model_name = config['MODEL']['name']
    # calculate label-wise mean latent vectors
    n_class = 10
    if config['DATA']['name'] == 'CIFAR100':
        n_class = 100
    total_mu = {i:[] for i in range(n_class)}
    total_logvar = {i:[] for i in range(n_class)}

    pbar = tqdm.tqdm(train_loader, desc='calculating mean latent vectors')
    with torch.no_grad():
        for batch_input in pbar:
            data = epochs.unpack_data(batch_input, device=device, model_type='cvae', is_train=True)
            x,y = data
            mu, logvar = model.encoder(x)

            for ci in range(n_class):
                idx = (y == ci)
                total_mu[ci].append(mu[idx])
                total_logvar[ci].append(logvar[idx])
    
    latent_dim = config['MODEL']['latent_dim']
    for ci in range(n_class):
        total_mu[ci] = torch.cat(total_mu[ci], dim=0)
        total_logvar[ci] = torch.cat(total_logvar[ci], dim=0)
        total_mu[ci] = total_mu[ci].mean(dim=0)
        #total_logvar[ci] = total_logvar[ci].mean(dim=0)
        total_logvar[ci] = torch.ones(latent_dim, dtype=torch.float32, device=device) * 0.5

    """
    # uncondition
    for ci in range(n_class):
        if config['DATA']['name'] == 'MNIST':
            total_mu[ci] = torch.zeros(4, dtype=torch.float32, device=device)
            total_logvar[ci] = torch.ones(4, dtype=torch.float32, device=device) * 0.5
        else:
            total_mu[ci] = torch.zeros(48, dtype=torch.float32, device=device)
            total_logvar[ci] = torch.ones(48, dtype=torch.float32, device=device) * 0.5
    """
    
    # generate samples per class
    num_samples = args.num_samples
    gen_x = []
    gen_y = []
    for ci in tqdm.tqdm(range(n_class), desc='generating samples'):
        mu = total_mu[ci].unsqueeze(0)
        logvar = total_logvar[ci].unsqueeze(0)
        pz = model.pz(mu, logvar)
        z = pz.rsample(torch.Size([num_samples]))

        if num_samples > 1000:
            zs = z.split(10)
            for sub_z in zs:
                x_hat, y = generate_data(model, sub_z, ci, model_name)
                if config['DATA']['name'] == 'MNIST':
                    x_hat = x_hat.squeeze(-1)
                gen_x.append(x_hat)
                gen_y.append(y)
        else:
            x_hat, y = generate_data(model, z, ci, model_name)
            if config['DATA']['name'] == 'MNIST':
                x_hat = x_hat.squeeze(-1)
            gen_x.append(x_hat)
            gen_y.append(y)

        # write to image
        N = 16
        K = 1
        img_path = os.path.join(save_path, f'gen_{ci}.png')
        try:
            xi = x_hat[:N].view(K,N, *x_hat.size()[1:]).transpose(0,1)
        except RuntimeError as e:
            print(e)
            import ipdb; ipdb.set_trace()

        if config['DATA']['name'] == 'MNIST':
            xi = xi.unsqueeze(2)
        else:
            xi = xi.permute(0,1,4,2,3)
        
        s = [make_grid(t, nrow=int(np.sqrt(K)), padding=0) for t in xi]
        logging.save_img(torch.stack(s), img_path, nrow=4)
    
    gen_x = torch.cat(gen_x, dim=0)
    gen_y = torch.cat(gen_y, dim=0)

    # save generated data
    save_dict = {'x': gen_x, 'y': gen_y}
    save_path = os.path.join(save_path, 'generated_data.pt')
    torch.save(save_dict, save_path)
    print(f"generated data saved to {save_path}")

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    print(config)
    main(config, args)