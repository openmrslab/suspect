import numpy


def svd_weighting(data):
    p, _, v = numpy.linalg.svd(data, full_matrices=False)
    channel_weights = p[:, 0].conjugate()

    # try some basic phase correction
    # in our truncation of the SVD to rank 1, we know that v[0] is our FID
    # use the first point of it to phase the signal
    phase_shift = numpy.angle(v[0, 0])

    return channel_weights * numpy.exp(-1j * phase_shift) / numpy.sum(numpy.abs(channel_weights))


def combine_channels(data, weights=None):
    if weights is None:
        weights = svd_weighting(data)
    weighted_data = weights.reshape((len(weights), 1)) * data
    combined_data = weighted_data.sum(axis=-2)
    return combined_data
