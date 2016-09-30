import suspect.basis

import numpy


def gaussian_window(t, params):
    """
    Calculates a Gaussian window function in the time domain which will broaden
    peaks in the frequency domain by params["line_broadening"] Hertz.

    Parameters
    ----------
    t : arange ndarray
        time axis
    params :

    Returns
    -------

    """
    window = suspect.basis.gaussian(t, 0, 0, params["line_broadening"])

    # the above gaussian function returns an area 1 fid, for a windowing
    # function we need to be area preserving (first point must be 1)
    return window / window[0]


def apodize(data, function, params):
    time_axis = data.time_axis()
    window = function(time_axis, params)
    return data.inherit(numpy.multiply(data, window))
