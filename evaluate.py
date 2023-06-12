# evaluation with FID and IS score
# IS case:
# python evaluate.py --config configs/cvae_mnist_clsmin.yml --metric is --gen_path saves/cvae_MNIST_CVAE_alpha_clsmin/generated_data.pt
# python evaluate.py --config configs/cvae_cifar10_clsmin.yml --metric is --gen_path saves/cvae_CIFAR10_CVAE_alpha_clsmin/generated_data.pt
# python evaluate.py --config configs/cvae_cifar100_clsmin.yml --metric is --gen_path saves/cvae_CIFAR100_CVAE_alpha_clsmin/generated_data.pt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.models.inception import inception_v3
from torch.nn.functional import adaptive_avg_pool2d

from scipy.stats import entropy
from scipy import linalg
import os
import argparse
import numpy as np
import yaml
from tqdm import tqdm

from mmdatasets.datautils import get_dataloader

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-modal VAE')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    parser.add_argument('--metric', type=str, default='is', help='evaluation metric. is or fid')
    parser.add_argument('--gen_path', type=str, default='', help='path of generated images')
    return parser.parse_args()

class GenDataset(Dataset):
    def __init__(self, data, eval_method):
        if eval_method == 'acc':
            self.x = data['x']
            self.y = data['y']
            self.num_samples = len(self.y)
        else:
            self.data = data['x']
            self.num_samples = len(self.data)
        self.eval_method = eval_method

    def __getitem__(self, index):
        if self.eval_method == 'acc':
            return self.x[index].detach(), self.y[index].detach()
        else:
            return self.data[index]

    def __len__(self):
        return self.num_samples

def upsample(data):
    return F.interpolate(data, scale_factor=2, mode='bilinear', align_corners=True)

def get_preds(loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch_input in tqdm(loader, desc='getting preds'):
            batch_input = batch_input.to(device)

            if batch_input.dim()==3:
                batch_input = batch_input.unsqueeze(1)
            else:
                batch_input = batch_input.permute(0,3,1,2)

            batch_input = upsample(batch_input)
            out = model(batch_input)
            preds.append(F.softmax(out).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds

def inception_score(config, gen_loader, model, device):
    gen_preds = get_preds(gen_loader, model, device)

    py = np.mean(gen_preds, axis=0) # p(y)
    
    scores = []
    for i in range(gen_preds.shape[0]):
        pyx = gen_preds[i]
        scores.append(entropy(pyx, py))
    scores = np.array(scores)
    return np.exp(np.mean(scores))

def get_fid_out(model, x):
    x = model.Conv2d_1a_3x3(x)
    # N x 32 x 149 x 149
    x = model.Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x = model.Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x = model.maxpool1(x)
    # N x 64 x 73 x 73
    x = model.Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x = model.Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x = model.maxpool2(x)
    # N x 192 x 35 x 35
    x = model.Mixed_5b(x)
    # N x 256 x 35 x 35
    x = model.Mixed_5c(x)
    # N x 288 x 35 x 35
    x = model.Mixed_5d(x)
    # N x 288 x 35 x 35
    x = model.Mixed_6a(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6b(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6c(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6d(x)
    # N x 768 x 17 x 17
    x = model.Mixed_6e(x)
    # N x 768 x 17 x 17
    x = model.Mixed_7a(x)
    # N x 1280 x 8 x 8
    x = model.Mixed_7b(x)
    # N x 2048 x 8 x 8
    x = model.Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = model.avgpool(x)
    return x

# forward w/o aux logits
def forward_inception(model, x):
    x = get_fid_out(model, x)
    x = model.dropout(x)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 2048
    x = model.fc(x)
    return x

def get_activation_stats(model, loader, info_path, device):
    if os.path.exists(info_path):
        data = np.load(info_path)
        mu, sigma = data['mu'], data['sigma']
    else:
        activations = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='calculating activations for fid'):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                if batch.dim()==3:
                    batch = batch.unsqueeze(1)
                elif batch.shape[-1] == 3:
                    batch = batch.permute(0,3,1,2)
                batch = upsample(batch)
                feat = get_fid_out(model, batch)
                feat = feat.squeeze(3).squeeze(2).detach().cpu().numpy()
                activations.append(feat)
        activations = np.concatenate(activations, axis=0)

        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        dic = {'mu': mu, 'sigma': sigma}
        np.savez(info_path, sigma)
    return mu, sigma

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def fid_score(config, train_loader, gen_loader, model, device):
    real_fid_info_path = f'fid_info_real_{config["DATA"]["name"]}.npz'
    gen_fid_info_path = f'fid_info_gen_{config["DATA"]["name"]}.npz'
    real_mu, real_sigma = get_activation_stats(model, train_loader, real_fid_info_path, device)
    gen_mu, gen_sigma = get_activation_stats(model, gen_loader, gen_fid_info_path, device)

    fid = calculate_fid(real_mu, real_sigma, gen_mu, gen_sigma)
    return fid

def finetune_inception(config, train_loader, test_loader, model, device, save=True, dual_train=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    epochs = 3
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'[{epoch}/{epochs}]finetuning')):
            data, target = data.to(device), target.to(device).long()
            if data.dim()==3:
                data = data.unsqueeze(1)
            elif data.shape[-1] == 3:
                data = data.permute(0,3,1,2)
            data = upsample(data)
            optimizer.zero_grad()
            output = forward_inception(model, data)
            loss = F.cross_entropy(output, target)
            loss.backward()
        
        if dual_train:
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc=f'[{epoch}/{epochs}]finetuning')):
                data, target = data.to(device), target.to(device).long()
                if data.dim()==3:
                    data = data.unsqueeze(1)
                elif data.shape[-1] == 3:
                    data = data.permute(0,3,1,2)
                data = upsample(data)
                optimizer.zero_grad()
                output = forward_inception(model, data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
        scheduler.step()
    
    # save model
    if save:
        torch.save(model.state_dict(), f"inception_{config['DATA']['name']}.pt")
    else:
        return model

def compute_accuracy(config, loader, model, device, debug=False):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for x,y in tqdm(loader):
            x,y = x.to(device), y.to(device)
            out = model(x)
            acc = (out.argmax(1)==y).float().sum()
            accuracy += acc.item()
    return accuracy/len(loader.dataset)

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    print(config)
    config['DATA']['batch_size'] = 48
    eval_method = args.metric

    # get gt data
    train_loader, test_loader = get_dataloader(config)

    # get generated data
    gen_data = torch.load(args.gen_path)
    gen_dataset = GenDataset(gen_data, eval_method)
    gen_loader = DataLoader(gen_dataset, batch_size=config['DATA']['batch_size'], 
                            shuffle=False, num_workers=config['DATA']['num_workers'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get inception model
    model = inception_v3(pretrained=True, transform_input=False)

    model.maxpool1 = nn.Identity()
    model.maxpool2 = nn.Identity()
    model.AuxLogits = None
    model.aux_logits = False

    dataset_name = config['DATA']['name']
    if dataset_name == 'MNIST':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(2048, 10)
    elif dataset_name == 'CIFAR10':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(2048, 10)
    elif dataset_name == 'CIFAR100':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.fc = nn.Linear(2048, 100)
    
    if not os.path.exists(f'inception_{dataset_name}.pt'):
        model = model.to(device)
        finetune_inception(config, train_loader, test_loader, model, device)
    model.load_state_dict(torch.load(f'inception_{dataset_name}.pt'))

    model = model.to(device)
    model.eval()

    if eval_method == 'fid':
        fid_score = fid_score(config, train_loader, gen_loader, model, device)
        print('FID score: {}'.format(fid_score))
    elif eval_method == 'is':
        is_score = inception_score(config, gen_loader, model, device)
        print('IS score: {}'.format(is_score))
    elif eval_method == 'acc':
        acc = compute_accuracy(config, test_loader, model, device)
        print('Accuracy on original test set: {}'.format(acc))

        # finetune on generated data
        model = finetune_inception(config, gen_loader, train_loader, model, device, save=False, dual_train=True)
        acc = compute_accuracy(config, test_loader, model, device, debug=True)
        print('Accuracy after finetuning with generated data: {}'.format(acc))

