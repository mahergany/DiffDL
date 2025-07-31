import numpy as np
import os
import jax
import jax.numpy as jnp
from . import utils
from . import inception
import functools
import scipy
import tempfile

def get_fid_through_path(source_images_path, generated_images_path, grayscale=True):

    rng = jax.random.PRNGKey(0)

    model = inception.InceptionV3(pretrained=True)
    params = model.init(rng, jnp.ones((1, 256, 256, 3)))

    apply_fn = jax.jit(functools.partial(model.apply, train=False))

    mu1, sigma1 = utils.compute_data_distribution_with_path(source_images_path,apply_fn=apply_fn, fn_params=params,isGenerated=False)
    mu2, sigma2 = utils.compute_data_distribution_with_path(generated_images_path,apply_fn=apply_fn, params=params,isGenerated=True)

    fid = compute_fid(mu1, sigma1, mu2, sigma2)

    return fid


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Taken from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_1d(sigma1)
    sigma2 = np.atleast_1d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


if __name__ == '__main__':
   get_fid_through_path()