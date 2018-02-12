import suspect

import numpy as np


def test_whiten():
    noise = suspect.MRSData(np.random.randn(32, 2048), 1e-3, 123)
    # make sure the noise has some correlations in it
    noise[0] += noise[1]
    noise[2] += 0.25 * noise[3]
    noise[4] += 0.9 * noise[5]
    # make sure the covariance matrix is not I
    cov_pre = np.cov(noise)
    np.testing.assert_raises(AssertionError, np.testing.assert_equal, cov_pre, np.eye(32))
    white_noise = suspect.processing.channel_combination.whiten(noise, noise)
    cov_post = np.cov(white_noise)
    np.testing.assert_almost_equal(cov_post, np.eye(32))

    # do the whitening specifying a length of noise
    white_noise = suspect.processing.channel_combination.whiten(noise, 2048)
    cov_post = np.cov(white_noise)
    np.testing.assert_almost_equal(cov_post, np.eye(32))
