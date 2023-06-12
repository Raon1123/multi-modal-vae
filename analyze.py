import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from skimage.transform import resize
from mmdatasets.datautils import get_dataloader

device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")
PATH = './saves0603/vae_CIFAR10_VAE_alpha.pt'
model = torch.load(PATH)
model.cuda()
config = './configs/cifar10.yaml'
train_loader, test_loader = get_dataloader(config)

#model.load_state_dict(torch.load(runPath + '/model.rar', **conversion_kwargs), strict=False)

class Latent_Classifier(nn.Module):
    def __init__(self, in_n, out_n):
        super(Latent_Classifier, self).__init__()
        self.mlp = nn.Linear(in_n, out_n)

    def forward(self, x):
        return self.mlp(x)

def classify_latents(epochs, option):
    model.eval()
    if '_' not in args.model:
        epochs *= 10  # account for the fact the mnist-svhn has more examples (roughly x10)
    classifier = Latent_Classifier(args.latent_dim, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_iters = len(train_loader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, data in enumerate(train_loader):
            # get the inputs
            x, targets = unpack_data_mlp(data, option)
            x, targets = x.to(device), targets.to(device)
            with torch.no_grad():
                qz_x_params = model.enc(x)
                zs = model.qz_x(*qz_x_params).rsample()
            optimizer.zero_grad()
            outputs = classifier(zs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (i + 1) % 1000 == 0:
                print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training, calculating test loss...')

    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, targets = unpack_data_mlp(data, option)
            x, targets = x.to(device), targets.to(device)
            qz_x_params = model.enc(x)
            zs = model.qz_x(*qz_x_params).rsample()
            outputs = classifier(zs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print('The classifier correctly classified {} out of {} examples. Accuracy: '
          '{:.2f}%'.format(correct, total, correct / total * 100))





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist', metavar='D', help = 'dataset name (default: mnist)' )
    parser.add_argument("--model", type=str, default='./saves0603/vae_CIRFAR10_VAE_alpha.pt', metavar='M')
    parser.add_argument('--save-dir', type=str, default="./eval_result", metavar='N', help='save directory of results')
    args = parser.parse_args()
    
    classify_latents(epochs=30)