import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import pandas as pd
from umap import UMAP

import utils.logging as logging
from utils.criteria import kl_divergence
from .vae import tensor_to_df, tensors_to_df, VAE

def one_hot(c, c_dim=10):
    if isinstance(c, int):
        return torch.eye(c_dim, dtype=torch.float32)[c]
    else:
        return torch.eye(c_dim, dtype=torch.float32, device=c.device)[c]

class CVAE(nn.Module):
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

        self.c_dim = params['c_dim']

        self.scaling_factor = 1.0

        if 'c_mask' in params:
            self.c_mask = params['c_mask']
        else:
            self.c_mask = 0

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

    def forward(self, x, c, K=1):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar, K)
        
        if c is None: # for test set
            c_extend = torch.zeros((K, x.size(0), self.c_dim), dtype=torch.float32, device=x.device)
        else:
            c_vec = one_hot(c, self.c_dim)
            c_extend = c_vec.expand(K,-1,-1)
            if self.c_mask > 0: # mask out some of the c_extend with given ratio c_mask
                mask = torch.rand(c_extend.shape[:-1], dtype=torch.float32, device=x.device)
                mask = mask > self.c_mask
                c_extend = c_extend * mask.unsqueeze(-1)
        
        x_hat = self.decoder(z, c_extend)
        return x_hat, z, mu, logvar
    
    def generate(self, N, c=None):
        self.eval()
        with torch.no_grad():
            pz = self.pz(*self.pz_params)
            latents = pz.rsample(torch.Size([N]))

            if c is None:
                # random sample c
                c_idxs = torch.randint(0, self.c_dim, (N,))
                c = one_hot(c_idxs, c_dim=self.c_dim).to(latents.device)
                c = c.unsqueeze(1)
            else:
                c_idxs = c
                c = one_hot(c, c_dim=self.c_dim).to(latents.device)

            samples = self.decoder(latents, c)
        return samples.view(-1, *samples.shape[-3:]), c_idxs
    
    def reconstruct(self, x, c=None):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            latents = self.reparameterize(mu, logvar)

            if c is None:
                c = torch.zeros((x.size(0), self.c_dim), dtype=torch.float32, device=x.device)
            else:
                c = one_hot(c, self.c_dim).to(latents.device)
            c = c.unsqueeze(0)
            
            recon = self.decoder(latents, c).view(-1, *x.shape[-3:])
        return recon
    
    def analyse(self, x, c=None, K=10):
        self.eval()
        with torch.no_grad():
            x_hat, zs, mu, logvar = self.forward(x, c, K)
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

class MNISTCondEncoder(nn.Module):
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
        emb = self.encoder(x.reshape(*x.size()[:-3], -1))
        mu = self.fc_mu(emb)
        logvar = self.fc_var(emb)
        logvar = F.softmax(logvar, dim=-1) * logvar.size(-1) + 1e-6
        return mu, logvar
    

class MNISTCondDecoder(nn.Module):
    def __init__(self, latent_dim, c_dim, hidden_layers, hidden_dim=400) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.c_dim = c_dim

        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim + c_dim, hidden_dim), nn.ReLU(True)))
        for _ in range(hidden_layers-1):
            modules.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True)))
        self.decoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dim, 784)

    def forward(self, z, c):
        cat_dim = 2 if z.dim()==3 else 1
        decoder_input = torch.cat((z, c), dim=cat_dim)
        p = self.fc(self.decoder(decoder_input))
        d = torch.sigmoid(p.view(*p.size()[:-1], 1, 28, 28))
        d = d.clamp(1e-6, 1-1e-6)

        return d

class MNISTCVAE(CVAE):
    def __init__(self, params):
        super().__init__(
            prior_dist=torch.distributions.Laplace,
            likelihood_dist=torch.distributions.Laplace,
            posterior_dist=torch.distributions.Laplace,
            encoder=MNISTCondEncoder(params['latent_dim'], params['hidden_layers']),
            decoder=MNISTCondDecoder(params['latent_dim'], params['c_dim'], params['hidden_layers']),
            params=params
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params['latent_dim']), requires_grad=False), # mu
            nn.Parameter(torch.ones(1, params['latent_dim']), requires_grad=params['learn_prior']) # logvar
        ])

        self.modelname = 'mnist_cvae'
        self.data_size = torch.tensor([1, 28, 28])
        self.scaling_factor = 1.0

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1) + 1e-6
    
    def generate(self, run_path, epoch, N=64, K=1, c=None, vis=True):
        gen_out = super().generate(N, c)
        samples = gen_out[0]
        gen_c_idxs = gen_out[1]

        if vis:
            samples = samples.cpu()
            gen_c_idxs = gen_c_idxs.cpu()
            samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)  # N x K x 1 x 28 x 28
            s = [make_grid(t, nrow=int(np.sqrt(K)), padding=0) for t in samples]
            logging.save_img(torch.stack(s), '{}/gen_samples_{:03d}.png'.format(run_path, epoch), n_row=8)
        else:
            return samples, gen_c_idxs


    def reconstruct(self, x, run_path, epoch):
        recon = super().reconstruct(x[:8])
        comp = torch.cat([x[:8], recon]).data.cpu()
        logging.save_img(comp, '{}/recon_{:03d}.png'.format(run_path, epoch))

    def analyse(self, x, run_path, epoch):
        z_emb, zsl, kls_df = super().analyse(x, K=10)
        labels = ['Prior', self.modelname.lower()]
        logging.plot_embeddings(z_emb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(run_path, epoch))
        logging.plot_kls_df(kls_df, '{}/kldistance_{:03d}.png'.format(run_path, epoch))

class CIFARCondEncoder(nn.Module):
    def __init__(self, latent_dim, hidden_layers=3, channels=[32,64,128]) -> None:
        super().__init__()

        prev_channel = 3
        channel = 32

        layers = []
        for l_i in range(hidden_layers):
            channel = channels[l_i]
            layers.append(nn.Conv2d(prev_channel, channel, 4, 2, 1, bias=True))
            layers.append(nn.ReLU(True))
            prev_channel = channel
        self.encoder = nn.Sequential(*layers)

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=True),
            nn.ReLU(True),
        )
        """
        # encoder output (128, 4, 4)

        self.c1 = nn.Conv2d(channels[-1], latent_dim, 4, 1, 0, bias=True)
        self.c2 = nn.Conv2d(channels[-1], latent_dim, 4, 1, 0, bias=True)

    def forward(self, x):
        enc = self.encoder(x)
        mu = self.c1(enc).squeeze()
        logvar = self.c2(enc).squeeze()
        logvar = F.softmax(logvar, dim=-1) * logvar.size(-1) + 1e-6

        return mu, logvar

class CIFARCondDecoder(nn.Module):
    def __init__(self, latent_dim, c_dim=10, hidden_layers=4, channels=[128, 64, 32, 3]) -> None:
        super().__init__()

        prev_channel = latent_dim + c_dim
        channel = 32

        layers = []
        for l_i in range(hidden_layers):
            channel = channels[l_i]
            if l_i == 0:
                layers.append(nn.ConvTranspose2d(prev_channel, channel, 4, 1, 0, bias=True))
            else:
                layers.append(nn.ConvTranspose2d(prev_channel, channel, 4, 2, 1, bias=True))
            if l_i + 1 == hidden_layers:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU(True))
            prev_channel = channel
        self.decoder = nn.Sequential(*layers)
        # decoder output (3, 32, 32)

    def forward(self, z, c):
        cat_dim = 2 if z.dim()==3 else 1
        decoder_input = torch.cat((z, c), dim=cat_dim).unsqueeze(-1).unsqueeze(-1)
        out = self.decoder(decoder_input.view(-1, *decoder_input.size()[-3:]))
        out = out.view(*decoder_input.size()[:-3], *out.size()[1:])
        return out


class CIFARCVAE(CVAE):
    def __init__(self, params, learn_prior=False) -> None:
        super().__init__(
            prior_dist=torch.distributions.Laplace,
            likelihood_dist=torch.distributions.Laplace,
            posterior_dist=torch.distributions.Laplace,
            encoder=CIFARCondEncoder(params['latent_dim'], params['enc_hidden_layers'], params['enc_channels']),
            decoder=CIFARCondDecoder(params['latent_dim'], params['c_dim'], params['dec_hidden_layers'], params['dec_channels']),
            params=params
        )

        self._pz_params = nn.ParameterList([
            nn.Parameter(torch.zeros(1, params['latent_dim']), requires_grad=False), # mu
            nn.Parameter(torch.ones(1, params['latent_dim']), requires_grad=params['learn_prior']) # logvar
        ])

        self.modelname = 'cifar_cvae'
        self.data_size = torch.tensor([3, 32, 32])
        self.scaling_factor = 1.0

    @property
    def pz_params(self):
        return self._pz_params[0], F.softmax(self._pz_params[1], dim=1) * self._pz_params[1].size(-1) + 1e-6
    
    def generate(self, run_path, epoch, N=64, K=1, c=None, vis=True):
        gen_out = super().generate(N, c)
        samples = gen_out[0]
        gen_c_idxs = gen_out[1]
        
        if vis:
            samples = samples.cpu()
            gen_c_idxs = gen_c_idxs.cpu()
            samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
            s = [make_grid(t, nrow=int(np.sqrt(K)), padding=0) for t in samples]
            logging.save_img(torch.stack(s), '{}/gen_samples_{:03d}.png'.format(run_path, epoch), n_row=8)
        else:
            return samples, gen_c_idxs
        
        return samples, gen_c_idxs

    def reconstruct(self, x, run_path, epoch):
        recon = super(CIFARCVAE, self).reconstruct(x[:8])
        comp = torch.cat([x[:8], recon]).data.cpu()
        logging.save_img(comp, '{}/recon_{:03d}.png'.format(run_path, epoch))

    def analyse(self, x, run_path, epoch):
        z_emb, zsl, kls_df = super().analyse(x, K=10)
        labels = ['Prior', self.modelname.lower()]
        logging.plot_embeddings(z_emb, zsl, labels, '{}/emb_umap_{:03d}.png'.format(run_path, epoch))
        logging.plot_kls_df(kls_df, '{}/kldistance_{:03d}.png'.format(run_path, epoch))