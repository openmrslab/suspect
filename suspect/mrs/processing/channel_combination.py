from suspect.mrs import MRSData

import numpy


def svd_weighting(data):
    p, _, _ = numpy.linalg.svd(data, full_matrices=False)
    channel_weights = p[:, 0].conjugate()

    return channel_weights
