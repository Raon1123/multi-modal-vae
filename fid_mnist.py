import warnings

import numpy
from keras.datasets import mnist
from keras.models import Model
from scipy import linalg

from DcGanBaseModel import DcGanBaseModel
from MnistClassifierModel06 import MnistClassifier06
from mnist.MnistModel02 import MnistModel02


class FrechetInceptionDistance:

    def __init__(self, real_activations: numpy.ndarray, verbose=False) -> None:
        self.real_activations = real_activations
        self.verbose = verbose

        self.real_mu = numpy.mean(real_activations, axis=0)
        self.real_sigma = numpy.cov(real_activations, rowvar=False)

    def compute_fid(self, fake_activations: numpy.ndarray):
        fake_mu = numpy.mean(fake_activations, axis=0)
        fake_sigma = numpy.cov(fake_activations, rowvar=False)
        fid = self.calculate_frechet_distance(fake_mu, fake_sigma, self.real_mu, self.real_sigma)
        return fid

    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        https://github.com/bioinf-jku/TTUR/blob/master/FIDvsINC/fid.py#L99-L148
        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1 : Numpy array containing the activations of the pool_3 layer of the
                 inception net ( like returned by the function 'get_predictions')
                 for generated samples.
        -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
                   on an representive data set.
        -- sigma1: The covariance matrix over activations of the pool_3 layer for
                   generated samples.
        -- sigma2: The covariance matrix over activations of the pool_3 layer,
                   precalcualted on an representive data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = numpy.atleast_1d(mu1)
        mu2 = numpy.atleast_1d(mu2)

        sigma1 = numpy.atleast_2d(sigma1)
        sigma2 = numpy.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not numpy.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = numpy.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if numpy.iscomplexobj(covmean):
            if not numpy.allclose(numpy.diagonal(covmean).imag, 0, atol=1e-3):
                m = numpy.max(numpy.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = numpy.trace(covmean)

        return diff.dot(diff) + numpy.trace(sigma1) + numpy.trace(sigma2) - 2 * tr_covmean


def compute_fid_score_for_gan(gan_model: DcGanBaseModel, classifier_model, layer_name, num_classes):
    # Define Feature Extracter
    feature_layer = Model(inputs=classifier_model.model.input,
                          outputs=classifier_model.model.get_layer(layer_name).output)

    # Compute Features for MNIST Dataset Images
    (x_train, _), _ = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    real_features = feature_layer.predict(x_train)
    fid = FrechetInceptionDistance(real_features, verbose=True)

    num_images = num_classes * 1000
    gen_images = gan_model.generate_images(num_images)
    fake_features = feature_layer.predict(gen_images)

    fid_score = fid.compute_fid(fake_features)
    return fid_score


def demo1():
    gan_model = MnistModel02(print_model_summary=False)
    gan_model.load_generator_model('../../Runs/01_MNIST/Model02/Run01/TrainedModels/generator_model_10000.h5')
    classifier_model = MnistClassifier06().load_model(
        '../../../../../DiscriminativeModels/01_MNIST_Classification/Runs/Run01/Trained_Models/MNIST_Model_10.h5')
    fid_score = compute_fid_score_for_gan(gan_model, classifier_model, 'dense_1', 10)
    print(fid_score)


if __name__ == '__main__':
    demo1()