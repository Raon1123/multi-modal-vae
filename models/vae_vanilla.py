import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid

import pandas as pd
from umap import UMAP

import utils.logging as logging
from utils.criteria import kl_divergence

def tensor_to_df(tensor, ax_names=None):
    assert tensor.ndim == 2, "Can only currently convert 2D tensors to dataframes"
    df = pd.DataFrame(data=tensor, columns=np.arange(tensor.shape[1]))
    return df.melt(value_vars=df.columns,
                   var_name=('variable' if ax_names is None else ax_names[0]),
                   value_name=('value' if ax_names is None else ax_names[1]))


def tensors_to_df(tensors, head=None, keys=None, ax_names=None):
    dfs = [tensor_to_df(tensor, ax_names=ax_names) for tensor in tensors]
    df = pd.concat(dfs, keys=(np.arange(len(tensors)) if keys is None else keys))
    df.reset_index(level=0, inplace=True)
    if head is not None:
        df.rename(columns={'level_0': head}, inplace=True)
    return df

class VAEVanilla(nn.Module):
    def __init__(self,
                 prior_dist,
                 likelihood_dist,
                 posterior_dist,
                 encoder,
                 decoder,
                 params) -> None:
        super().__init__()

        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist

        self.encoder = encoder
        self.decoder = decoder

        self.modelname = None
        self.params = params

        self._pz_params = None
        self._qz_x_params = None

        self.scaling_factor = 1.0

    @property
    def pz_params(self):
        return self._pz_params
    
    @property
    def qz_x_params(self):
        if self._qz_x_params is None:
            raise ValueError('qz_x parameters are not set.')
        return self._qz_x_params
    
    def reparameterize(self, mu, logvar, K=1):
        if isinstance(self.qz_x, torch.distributions.normal.Normal):
            std = torch.exp(0.5 * logvar)
        else:
            std = logvar
        z = self.qz_x(mu, std).rsample(torch.Size([K]))
        return z

    def forward(self, x, K=1, **args):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, K)
        x_hat = self.decoder(z)
        return x_hat, z, mu, logvar
    
    def generate(self, N):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))
            samples = self.decoder(latents)
        return samples.view(-1, *samples.shape[-3:])
    
    def reconstruct(self, x):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            latents = self.reparameterize(mu, logvar)
            #qz_x = self.qz_x(mu, logvar)
            #latents = qz_x.rsample()
            recon = self.decoder(latents).view(-1, *x.shape[-3:])
        return recon
    
    def analyse(self, x, K):
        self.eval()
        with torch.no_grad():
            x_hat, zs, mu, logvar = self.forward(x, K)
            pz = self.pz(*self.pz_params)
            qz_x = self.qz_x(mu, logvar)
            zss = [pz.sample(torch.Size([K, x.size(0)])).view(-1, pz.batch_shape[-1]),
                   zs.view(-1, zs.size(-1))]
            zsl = [torch.zeros(zs.size(0)).fill_(i) for i, zs in enumerate(zss)]
            kls_df = tensors_to_df(
                [kl_divergence(qz_x, pz).cpu().numpy()],
                head='KL',
                keys=[r'KL$(q(z|x)\,||\,p(z))$'],
                ax_names=['Dimensions', r'KL$(q\,||\,p)$']
            )
        
        ret = UMAP(metric='euclidean',
                   n_neighbors=5,
                   transform_seed=torch.initial_seed())
        ret = ret.fit_transform(torch.cat(zss, dim=0).cpu().numpy())

        return ret, torch.cat(zsl, 0).cpu().numpy(), kls_df
    

class MNISTEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_layers=1, hidden_dim=400) -> None:
        super().__init__()
        
        modules = []
        modules.append(nn.Sequential(nn.Linear(784, hidden_dim), nn.ReLU(True)))
        for _ in range(hidden_layers-1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)))
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        emb = self.encoder(x.view(*x.size()[:-3], -1))
        mu = self.fc_mu(emb)
        logvar = self.fc_var(emb)
        logvar = F.softmax(logvar, dim=-1) * logvar.size(-1) + 1e-6
        return mu, logvar
    

class MNISTDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_layers, hidden_dim=400) -> None:
        super().__init__()

        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        for _ in range(hidden_layers-1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)))
        self.decoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dim, 784)

    def forward(self, z):
        p = self.fc(self.decoder(z))
        d = torch.sigmoid(p.view(*p.size()[:-1], 1, 28, 28))
        d = d.clamp(1e-6, 1-1e-6)
        return d
    

class MNISTVAEVanilla(VAEVanilla):
    def __init__(self, params):
        super().__init__(
            prior_dist=torch.distributions.Laplace,
            likelihood_dist=torch.distributions.Laplace,
            posterior_dist=torch.distributions.Laplace,
            encoder=MNISTEncoder(params['latent_dim'], params['hidden_layers']),
            decoder=MNISTDecoder(params['latent_dim'], params['hidden_layers']),
            params=params
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params['latent_dim']), requires_grad=False), # mu
            nn.Parameter(torch.ones(1, params['latent_dim']), requires_grad=params['learn_prior']) # logvar
        ])

        self.modelname = 'mnist_vae'
        self.data_size = torch.tensor([1, 28, 28])
        self.scaling_factor = 1.0

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1) + 1e-6
    
    def generate(self, N=64, epoch=0, run_path='./'):
        K = 1
        rowsamples = super().generate(N).cpu()
        #samples = rowsamples.view(K, N, *rowsamples.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
        #s = [make_grid(t, nrow=int(np.sqrt(K)), padding=0) for t in samples]
        #logging.save_img(torch.stack(s), '{}/gen_samples_{:03d}.png'.format(run_path, epoch), nrow=8)
        return rowsamples

    def reconstruct(self, x, run_path, epoch):
        recon = super().reconstruct(x[:8])
        comp = torch.cat([x[:8], recon]).data.cpu()
        logging.save_img(comp, '{}/recon_{:03d}.png'.format(run_path, epoch))

    def analyse(self, x, run_path, epoch):
        z_emb, zsl, kls_df = super().analyse(x, K=10)
        labels = ['Prior', self.modelname.lower()]
        logging.plot_embeddings(z_emb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(run_path, epoch))
        logging.plot_kls_df(kls_df, '{}/kldistance_{:03d}.png'.format(run_path, epoch))

class CIFAREncoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()

        # input shape: 3 x 32 x 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.ReLU(True),
        )
        # encoder output (128, 4, 4)

        self.c1 = nn.Conv2d(128, latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(128, latent_dim, 4, 1, 0, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        mu = self.c1(enc).squeeze()
        logvar = self.c2(enc).squeeze()
        logvar = F.softmax(logvar, dim=-1) * logvar.size(-1) + 1e-6
        return mu, logvar


class CIFARDecoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()

        # input shape (latent_dim, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, 1, 0, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=True),
            nn.Sigmoid()
        )
        # decoder output (3, 32, 32)

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(z.view(-1, *z.size()[-3:]))
        out = out.view(*z.size()[:-3], *out.size()[1:])
        return out


class CIFARVAEVanilla(VAEVanilla):
    def __init__(self, params, learn_prior=False) -> None:
        super().__init__(
            prior_dist=torch.distributions.Laplace,
            likelihood_dist=torch.distributions.Laplace,
            posterior_dist=torch.distributions.Laplace,
            #prior_dist=torch.distributions.normal.Normal,
            #likelihood_dist=torch.distributions.normal.Normal,
            #posterior_dist=torch.distributions.normal.Normal,
            encoder=CIFAREncoder(params['latent_dim']),
            decoder=CIFARDecoder(params['latent_dim']),
            params=params
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params['latent_dim']), requires_grad=False), # mu
            nn.Parameter(torch.ones(1, params['latent_dim']), requires_grad=params['learn_prior']) # logvar
        ])

        self.modelname = 'cifar_vae'
        self.data_size = torch.tensor([3, 32, 32])
        self.scaling_factor = 1.0

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1) + 1e-6
    
    def generate(self, N=64, run_path='./', epoch=0):
        K = 1
        rowsamples = super().generate(N).cpu()

        #samples = rowsamples.view(K, N, *rowsamples.size()[1:]).transpose(0, 1)
        #s = [make_grid(t, nrow=int(np.sqrt(K)), padding=0) for t in samples]
        #logging.save_img(torch.stack(s), '{}/gen_samples_{:03d}.png'.format(run_path, epoch), nrow=8)
        
        return rowsamples

    def reconstruct(self, x, run_path, epoch):
        recon = super().reconstruct(x[:8])
        comp = torch.cat([x[:8], recon]).data.cpu()
        logging.save_img(comp, '{}/recon_{:03d}.png'.format(run_path, epoch))

    def analyse(self, x, run_path, epoch):
        z_emb, zsl, kls_df = super().analyse(x, K=10)
        labels = ['Prior', self.modelname.lower()]
        logging.plot_embeddings(z_emb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(run_path, epoch))
        logging.plot_kls_df(kls_df, '{}/kldistance_{:03d}.png'.format(run_path, epoch))