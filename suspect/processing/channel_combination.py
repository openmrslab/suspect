import numpy


def svd_weighting(data, axis=-2):

    # the data shape that we require is 2D with channels as the zeroth
    # dimension, the optional axis argument is a convenience to modify
    # the array to fit that profile
    if axis is not None:
        num_channels = data.shape[axis]
        data = numpy.moveaxis(data, axis, 0).reshape(num_channels, -1)

    p, _, v = numpy.linalg.svd(data, full_matrices=False)
    channel_weights = p[:, 0].conjugate()

    # try some basic phase correction
    # in our truncation of the SVD to rank 1, we know that v[0] is our FID
    # use the first point of it to phase the signal
    phase_shift = numpy.angle(v[0, 0])

    return channel_weights * numpy.exp(-1j * phase_shift) / numpy.sum(numpy.abs(channel_weights))


def whiten(data, noise=100):
    """Calculates and applies a whitening transform to remove any correlations
    between channels. If a separate noise signal is supplied, the transform is
    calculated from that, otherwise the last `noise` points of the data ADC are
    used.

    Parameters
    ----------
    data : MRSData
        The data to be whitened.
    noise : arraylike, int
        
    Returns
    -------
    MRSData
        The whitened data.
    """
    if numpy.isscalar(noise):
        data_noise = data[..., -noise:]
        # reshape the noise to put channels at the first index
        # and coalesce all other indices
        data_noise = numpy.moveaxis(data_noise, -2, 0).reshape((data.shape[-2], -1))
        # remove all zeros from the noise (probably uncollected data)
        data_noise = data_noise[:, data_noise[0] != 0]
    else:
        data_noise = noise

    # calculate the noise covariance
    cov = numpy.cov(data_noise)
    # do an eigenvalue decomposition and form the scaling matrix
    u, d, v = numpy.linalg.svd(cov)
    w = numpy.dot(u, numpy.diag(numpy.sqrt(1 / d)))
    # apply the transform to the data
    return data.inherit(w.T.conj() @ data)


def combine_channels(data, weights=None, axis=-2):
    if weights is None:
        weights = svd_weighting(data, axis)
    weighted_data = weights.reshape((len(weights), 1)) * data
    combined_data = weighted_data.sum(axis=axis)
    return combined_data
