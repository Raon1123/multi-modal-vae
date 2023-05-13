import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

from umap import UMAP

import utils.logging as logging
from utils.criteria import kl_divergence
from models.modelutils import tensors_to_df
from models.vae import VAE

class MMVAE(nn.Module):
    def __init__(self,
                 prior_dist,
                 params,
                 *vaes) -> None:
        
        super(MMVAE, self).__init__()

        self.pz = prior_dist
        self.vaes = nn.ModuleList([vae(params) for vae in vaes])

        self.modulename = None
        self.params = params

        self._pz_params = None

    @property
    def pz_params(self):
        return self._pz_params
    
    def forward(self, x, K=1):
        qz_xs, zss = [], []

        px_zs = [[None for _ in range(len(self.vaes))] for _ in range(len(self.vaes))]

        for m, vae in enumerate(self.vaes):
            qz_x, px_z, zs = vae(x[m], K=K)
            qz_xs.append(qz_x)
            zss.append(zs)
            px_zs[m][m] = px_z

        for e, zs in enumerate(zss):
            for d, vae in enumerate(self.vaes):
                if e != d:
                    px_zs[e][d] = vae.px_z(*vae.decoder(zs[d]))

        return qz_xs, px_zs, zss
    
    def generate(self, N):
        self.eval()
        with torch.no_grad():
            x = []
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            for d, vae in enumerate(self.vaes):
                px_z = vae.px_z(*vae.decoder(latents))
                x.append(px_z.mean.view(-1, *px_z.mean.size()[2:]))

        return x
    
    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            _, px_zs, _ = self.forward(x)
            recons = [[px_z.mean for px_z in r] for r in px_zs]

        return recons
    
    def analyse(self, x, K):
        self.eval()
        with torch.no_grad():
            qz_x, _, zss = self.forward(x, K)
            pz = self.pz(*self.pz_params)
            zss = [pz.sample(torch.Size([K, x[0].size(0)])).view(-1, pz.batch_shape[-1]), 
                   *[zs.view(-1, zs.size(-1)) for zs in zss]]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [*[kl_divergence(qz_x, pz).cpu().numpy() for qz_x in qz_xs],
                 *[0.5 * (kl_divergence(p, q) + kl_divergence(q, p)).cpu().numpy()
                   for p, q in combinations(qz_xs, 2)]],
                head='KL',
                keys=[*[r'KL$(q(z|x_{})\,||\,p(z))$'.format(i) for i in range(len(qz_xs))],
                      *[r'J$(q(z|x_{})\,||\,q(z|x_{}))$'.format(i, j)
                        for i, j in combinations(range(len(qz_xs)), 2)]],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        
        ret = UMAP(metric='euclidean',
                   n_neighbors=5,
                   transform_seed=torch.initial_seed())
        ret = ret.fit_transform(torch.cat(zss, dim=0).cpu().numpy())

        return ret, torch.cat(zsl, 0).cpu().numpy(), kls_df
            