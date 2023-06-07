import math

import torch
import numpy as np
import torch.nn.functional as F

"""
The most implementation of criteria from MMVAE official implementation
"""

def kl_divergence(d1, d2, K=100):
    if (type(d1), type(d2)) in torch.distributions.kl._KL_REGISTRY:
        return torch.distributions.kl_divergence(d1, d2)
    else:
        samples = d1.rsample(torch.Size([K]))
        return (d1.log_prob(samples) - d2.log_prob(samples)).mean(0)


def get_criteria(config):
    model_name = config['MODEL']['name'].lower()
    if model_name == 'vae':
        criteria = elbo
        t_criteria = iwae
    elif model_name == 'vaevanilla':
        criteria = elbo_vanilla
        t_criteria = iwae_vanilla
    elif model_name == 'cvae':
        criteria = cvae_objective_wrapper
        t_criteria = iwae_c
    else:
        raise NotImplementedError('Model criteria not implemented.')

    return criteria, t_criteria

# https://github.com/pytorch/examples/blob/main/vae/main.py
def elbo_vanilla(model, x, K=1, **args):
    x_hat, z, mu, logvar = model(x, K)
    
    x_expand = x.expand(x_hat.size(0), *x.size())
    bce_loss = F.binary_cross_entropy(x_hat, x_expand, reduction='sum') * model.scaling_factor

    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = (bce_loss + kld_loss) / x.size(0)
    return total_loss

def iwae_vanilla(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)

    lw_list = []
    for _x in x.split(S):
        x_hat, zs, mu, logvar = model(_x, K)

        lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)

        _x_expand = _x.expand(x_hat.size(0), *_x.size())
        lpx_z = F.binary_cross_entropy(x_hat, _x_expand, reduction='none') * model.scaling_factor
        lpx_z = lpx_z.view(*zs.shape[:2], -1)

        if isinstance(model.qz_x, torch.distributions.normal.Normal):
            std = torch.exp(0.5 * logvar)
        else:
            std = logvar
        qz_x = model.qz_x(mu, std)
        lqz_x = qz_x.log_prob(zs).sum(-1)
        lw_list.append(lpz + lpx_z.sum(-1) - lqz_x)

    lw = torch.cat(lw_list, 1)  # concat on batch
    #return log_mean_exp(lw).sum()
    return log_mean_exp(lw).sum() / x.size(0)

# aux_objective: auxiliary objective loss type. 'cls' for classification, 'entropy' for Shannon entropy
def cvae_objective_wrapper(model, data, K=1, aux_objective='cls_max', classifier=None):
    x, c = data
    # elbo
    qz_x, px_z, _ = model(x, c, K)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    elbo_loss = -(lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()

    # auxiliary objective
    aux_loss = torch.zeros(1, dtype=torch.float32, device=x.device)
    if aux_objective != '' and classifier is not None:
        samples = px_z.sample(torch.Size([K]))
        samples = samples.view(-1, *samples.size()[3:])
        
        with torch.no_grad():
            cls_output = classifier(samples)
            logit = F.softmax(cls_output, dim=1)
        if aux_objective=='cls_min': # cross entropy minimization
            target = c.repeat(K).repeat(K)
            aux_loss = aux_loss + F.cross_entropy(logit, target, reduction='sum')
        elif aux_objective=='cls_max': # cross entropy maximization
            target = c.repeat(K).repeat(K)
            aux_loss =  aux_loss - F.cross_entropy(logit, target, reduction='sum')
        elif aux_objective=='entropy': # entropy maximization
            min_val = torch.e # minimum bound is -1/e*c_dim
            aux_loss = aux_loss + min_val + torch.sum(logit * torch.log(logit + 1e-8), dim=1).mean() / model.c_dim
    
    total_loss = elbo_loss + aux_loss

    return total_loss

def elbo(model, x, K=1, **args):
    qz_x, px_z, _ = model(x, K)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return -(lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()

# helper to vectorise computation
def compute_microbatch_split(x, K):
    """ Checks if batch needs to be broken down further to fit in memory. """
    B = x[0].size(0) if is_multidata(x) else x.size(0)
    S = sum([1.0 / (K * np.prod(_x.size()[1:])) for _x in x]) if is_multidata(x) \
        else 1.0 / (K * np.prod(x.size()[1:]))
    S = int(1e8 * S)  # float heuristic for 12Gb cuda memory
    assert (S > 0), "Cannot fit individual data in memory, consider smaller K"
    return min(B, S)


def _iwae(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1) - lqz_x


def iwae(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()

def _iwae_c(model, x, K):
    """IWAE estimate for log p_\theta(x) -- fully vectorised."""
    qz_x, px_z, zs = model(x, None, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    lqz_x = qz_x.log_prob(zs).sum(-1)
    return lpz + lpx_z.sum(-1) - lqz_x

def iwae_c(model, x, K):
    """Computes an importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw = torch.cat([_iwae_c(model, _x, K) for _x in x.split(S)], 1)  # concat on batch
    return log_mean_exp(lw).sum()

def _dreg(model, x, K):
    """DREG estimate for log p_\theta(x) -- fully vectorised."""
    _, px_z, zs = model(x, K)
    lpz = model.pz(*model.pz_params).log_prob(zs).sum(-1)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    qz_x = model.qz_x(*[p.detach() for p in model.qz_x_params])  # stop-grad for \phi
    lqz_x = qz_x.log_prob(zs).sum(-1)
    lw = lpz + lpx_z.sum(-1) - lqz_x
    return lw, zs


def dreg(model, x, K, regs=None):
    """Computes a doubly-reparameterised importance-weighted ELBO estimate for log p_\theta(x)
    Iterates over the batch as necessary.
    """
    S = compute_microbatch_split(x, K)
    lw, zs = zip(*[_dreg(model, _x, K) for _x in x.split(S)])
    lw = torch.cat(lw, 1)  # concat on batch
    zs = torch.cat(zs, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zs.requires_grad:
            zs.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()

def is_multidata(dataB):
    return isinstance(dataB, list) or isinstance(dataB, tuple)

def log_mean_exp(value, dim=0, keepdim=False):
    return torch.logsumexp(value, dim, keepdim=keepdim) - math.log(value.size(dim))