import numpy


def _pad(input_signal, length, average=10):
    """
    Helper function which increases the length of an input signal. The original
    is inserted at the centre of the new signal and the extra values are set to
    the average of the first and last parts of the original, respectively.
    :param input_signal:
    :param length:
    :param average:
    :return:
    """
    padded_input_signal = numpy.zeros(length)
    start_offset = (len(padded_input_signal) - len(input_signal)) / 2.
    padded_input_signal[:start_offset] = numpy.average(input_signal[0:average])
    padded_input_signal[start_offset:(start_offset + len(input_signal))] = input_signal[:]
    padded_input_signal[(start_offset + len(input_signal)):] = numpy.average(input_signal[-average:])
    return padded_input_signal


def sliding_gaussian(input_signal, params):
    window_width = params["window_width"]
    window = numpy.linspace(-3, 3, window_width)
    window = numpy.exp(-window**2)
    window /= numpy.sum(window)
    # pad the signal to cover half the window width on each side
    padded_input = _pad(input_signal, len(input_signal) + window_width - 1)
    result = numpy.zeros(len(input_signal))
    for i in range(len(input_signal)):
        result[i] = numpy.dot(window, padded_input[i:(i + window_width)])
    return result
