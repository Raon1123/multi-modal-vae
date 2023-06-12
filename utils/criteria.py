import math

import torch
import numpy as np
<<<<<<< HEAD
import torch.nn.functional as F
=======
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89

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
<<<<<<< HEAD
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
=======
    if model_name == 'mmvae':
        criteria = m_elbo
        t_criteria = m_iwae
    elif model_name == 'vae':
        criteria = elbo
        t_criteria = iwae

    return criteria, t_criteria


def elbo(model, x, K=1):
    qz_x, px_z, _ = model(x, K)
    lpx_z = px_z.log_prob(x).view(*px_z.batch_shape[:2], -1) * model.scaling_factor
    kld = kl_divergence(qz_x, model.pz(*model.pz_params))
    return (lpx_z.sum(-1) - kld.sum(-1)).mean(0).sum()


# multi-modal variants
def m_elbo_naive(model, x, K=1):
    """Computes E_{p(x)}[ELBO] for multi-modal vae --- NOT EXPOSED"""
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d, px_z in enumerate(px_zs[r]):
            lpx_z = px_z.log_prob(x[d]) * model.vaes[d].scaling_factor
            lpx_zs.append(lpx_z.view(*px_z.batch_shape[:2], -1).sum(-1))
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()


def m_elbo(model, x, K=1):
    """Computes importance-sampled m_elbo (in notes3) for multi-modal vae """
    qz_xs, px_zs, zss = model(x)
    lpx_zs, klds = [], []
    for r, qz_x in enumerate(qz_xs):
        kld = kl_divergence(qz_x, model.pz(*model.pz_params))
        klds.append(kld.sum(-1))
        for d in range(len(px_zs)):
            lpx_z = px_zs[d][d].log_prob(x[d]).view(*px_zs[d][d].batch_shape[:2], -1)
            lpx_z = (lpx_z * model.vaes[d].scaling_factor).sum(-1)
            if d == r:
                lwt = torch.tensor(0.0)
            else:
                zs = zss[d].detach()
                lwt = (qz_x.log_prob(zs) - qz_xs[d].log_prob(zs).detach()).sum(-1)
            lpx_zs.append(lwt.exp() * lpx_z)
    obj = (1 / len(model.vaes)) * (torch.stack(lpx_zs).sum(0) - torch.stack(klds).sum(0))
    return obj.mean(0).sum()


def _m_iwae(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].scaling_factor).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 1)  # concat on batch
    return log_mean_exp(lw).sum()


def _m_iwae_looser(model, x, K=1):
    """IWAE estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    lws = []
    for r, qz_x in enumerate(qz_xs):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x.log_prob(zss[r]).sum(-1) for qz_x in qz_xs]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].scaling_factor).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws)  # (n_modality * n_samples) x batch_size, batch_size


def m_iwae_looser(model, x, K=1):
    """Computes iwae estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw = [_m_iwae_looser(model, _x, K) for _x in x_split]
    lw = torch.cat(lw, 2)  # concat on batch
    return log_mean_exp(lw, dim=1).mean(0).sum()


def _m_dreg(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised"""
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].scaling_factor).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.cat(lws), torch.cat(zss)


def m_dreg(model, x, K=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 1)  # concat on batch
    zss = torch.cat(zss, 1)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 0, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).sum()


def _m_dreg_looser(model, x, K=1):
    """DERG estimate for log p_\theta(x) for multi-modal vae -- fully vectorised
    This version is the looser bound---with the average over modalities outside the log
    """
    qz_xs, px_zs, zss = model(x, K)
    qz_xs_ = [vae.qz_x(*[p.detach() for p in vae.qz_x_params]) for vae in model.vaes]
    lws = []
    for r, vae in enumerate(model.vaes):
        lpz = model.pz(*model.pz_params).log_prob(zss[r]).sum(-1)
        lqz_x = log_mean_exp(torch.stack([qz_x_.log_prob(zss[r]).sum(-1) for qz_x_ in qz_xs_]))
        lpx_z = [px_z.log_prob(x[d]).view(*px_z.batch_shape[:2], -1)
                     .mul(model.vaes[d].scaling_factor).sum(-1)
                 for d, px_z in enumerate(px_zs[r])]
        lpx_z = torch.stack(lpx_z).sum(0)
        lw = lpz + lpx_z - lqz_x
        lws.append(lw)
    return torch.stack(lws), torch.stack(zss)


def m_dreg_looser(model, x, K=1):
    """Computes dreg estimate for log p_\theta(x) for multi-modal vae
    This version is the looser bound---with the average over modalities outside the log
    """
    S = compute_microbatch_split(x, K)
    x_split = zip(*[_x.split(S) for _x in x])
    lw, zss = zip(*[_m_dreg_looser(model, _x, K) for _x in x_split])
    lw = torch.cat(lw, 2)  # concat on batch
    zss = torch.cat(zss, 2)  # concat on batch
    with torch.no_grad():
        grad_wt = (lw - torch.logsumexp(lw, 1, keepdim=True)).exp()
        if zss.requires_grad:
            zss.register_hook(lambda grad: grad_wt.unsqueeze(-1) * grad)
    return (grad_wt * lw).mean(0).sum()

>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89

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

<<<<<<< HEAD
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
=======
>>>>>>> 85b98ae1e474031f9bcc1311fc33bc35cf5c2b89

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