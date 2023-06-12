import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize

from main import load_config, parse_args
from models.modelutils import get_model, get_optimizer, get_classifier
from PIL import Image


## https://www.kaggle.com/code/ibtesama/gan-in-pytorch-with-fid
#def FID
def calculate_activation_statistics(img, model, batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(img), dims))
    
    if cuda:
        batch=img.cuda()
    else:
        batch=img
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = np.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = np.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    
def FID_score(gt_img,gen_img,model):
     mu_1,std_1 = calculate_activation_statistics(gt_img,model,cuda=True)
     mu_2,std_2 = calculate_activation_statistics(gen_img,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     
     return fid_value


## https://www.kaggle.com/code/aprilryan/inception-score 
# calculate inception score for cifar-10 in Keras

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)
 
# assumes images have any shape and pixels in [0,255]
def Inception_Score(model, images, n_split=10, eps=1E-16):
    #model = InceptionV3()
    scores = list()
    n_part = math.floor(images.shape[0] / n_split)
    for i in range(n_split):
        # retrieve images
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        subset = subset.astype('float32')
        # scale images to the required size
        subset = scale_images(subset, (299,299,3))
        # pre-process images, scale to [-1,1]
        subset = preprocess_input(subset)
        # predict p(y|x)
        p_yx = model.predict(subset)
        # calculate p(y)
        p_y = np.expand_dims(np.mean(p_yx, axis=0), 0)
        
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = np.mean(sum_kl_d)
        is_score = np.exp(avg_kl_d)
        scores.append(is_score)
    # average across images
    is_avg, is_std = np.mean(scores), np.std(scores)
    return is_avg, is_std

"""
# load cifar10 images
(images, _), (_, _) = cifar10.load_data()
# shuffle images
np.random.shuffle(images)
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = Inception_Score(images)
print('score', is_avg, is_std)
""" 


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dataset", type=str, default='mnist', metavar='D', help = 'dataset name (default: mnist)')
    #parser.add_argument("--predict_dir", type=str, default='./saves0604')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file.')
    #parser.add_argument("--gt_dir", type=str, default='./multi-modal-vae/ground_truth', metavar='G')
    args = parser.parse_args()
    config = load_config(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config)
    model.to(device)
    
    #fid_list = []
    
    """
    for i in range(10000):
        gen_img_nm = 'gen_samples_{:05d}.png'.format(i+1)
        #gt_img_nm = ''
        gen_img = Image.open(os.path.join('./saves0604', gen_img_nm))
        #gt_img = Image.open(os.path.join(args.gt_dir, gt_img_nm))
        
        #fid = FID_score(gt_img, gen_img, model)
        #fid_list.append(fid)
        import ipdb; ipdb.set_trace()
        ic = Inception_Score(model, gen_img)
        ic_list.append(ic)
    """
    
    
    
    #fid_score = np.mean(fid_list)
    #ic_score = np.mean(ic_list)
        