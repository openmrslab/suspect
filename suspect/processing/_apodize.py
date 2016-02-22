import numpy


def gaussian_window(t, params):
    """
    Calculates a Gaussian window function in the time domain which will broaden
    peaks in the frequency domain by params["line_broadening"] Hertz.
    :param t:
    :param params:
    :return:
    """
    window = numpy.sqrt(numpy.pi / 4 / numpy.log(2)) / params["line_broadening"] * numpy.exp(- t ** 2 / 4 * numpy.pi ** 2 / numpy.log(2) * params["line_broadening"] ** 2)
    return window / numpy.sum(window)


def apodize(data, function, params):
    time_axis = data.time_axis()
    window = function(time_axis, params)
    return data.inherit(numpy.multiply(data, window))
