import numpy


def _pad(input_signal, length, average=10):
    """Helper function which increases the length of an input signal.

    The original is inserted at the centre of the new signal and the extra values are set to
    the average of the first and last parts of the original, respectively.

    Parameters
    ----------
    input_signal:
        the signal to be padded
    length: int
        the length of the padded signal
    average: int
        the number of points at the beginning/end of the signal that are averaged to calculate the padded value

    Returns
    -------
    padded_input_signal : ndarray

    """
    padded_input_signal = numpy.zeros(length, input_signal.dtype)
    start_offset = int((len(padded_input_signal) - len(input_signal)) / 2)
    padded_input_signal[:start_offset] = numpy.average(input_signal[0:average])
    padded_input_signal[start_offset:(start_offset + len(input_signal))] = input_signal[:]
    padded_input_signal[(start_offset + len(input_signal)):] = numpy.average(input_signal[-average:])
    return padded_input_signal


def sliding_window(input_signal, window_width):
    window = numpy.ones(window_width)
    window /= numpy.sum(window)
    # pad the signal to cover half the window width on each side
    padded_input = _pad(input_signal, len(input_signal) + window_width - 1)
    result = numpy.zeros_like(input_signal)
    for i in range(len(input_signal)):
        result[i] = numpy.dot(window, padded_input[i:(i + window_width)])
    return result


def sliding_gaussian(input_signal, window_width):
    window = numpy.linspace(-3, 3, window_width)
    window = numpy.exp(-window**2)
    window /= numpy.sum(window)
    # pad the signal to cover half the window width on each side
    padded_input = _pad(input_signal, len(input_signal) + window_width - 1)
    result = numpy.zeros_like(input_signal)
    for i in range(len(input_signal)):
        result[i] = numpy.dot(window, padded_input[i:(i + window_width)])
    return result


def sift(input_signal, threshold):
    ft = numpy.fft.fft(input_signal)
    ft[numpy.absolute(ft) < threshold] = 0.0
    sifted = numpy.fft.ifft(ft)
    # applying SIFT to real data should also return real data, but casting to
    # a real type raises a ComplexWarning if we don't do this first
    if numpy.isrealobj(input_signal):
        sifted = sifted.real
    return sifted.astype(input_signal.dtype)


def svd(input_signal, rank):
    matrix_width = int(len(input_signal) / 2)
    matrix_height = len(input_signal) - matrix_width + 1
    hankel_matrix = numpy.zeros((matrix_width, matrix_height), input_signal.dtype)
    for i in range(matrix_height):
        hankel_matrix[:, i] = input_signal[i:(i + matrix_width)]
    # perform the singular value decomposition
    U, s, V = numpy.linalg.svd(numpy.matrix(hankel_matrix), full_matrices=False)

    s[rank:] = 0.0

    recon = U * numpy.diag(s) * V
    result = numpy.zeros_like(input_signal)
    for i in range(len(input_signal)):
        count = 0
        for j in range(matrix_height):
            x_offset = i - j
            if 0 <= x_offset < matrix_width:
                count += 1
                result[i] += recon[x_offset, j]
        result[i] /= count
    return result


def spline(input_signal, num_splines, spline_order):
    # input signal  has to be a multiple of num_splines
    padded_input_signal = _pad(input_signal, int(numpy.ceil(len(input_signal) / float(num_splines))) * num_splines)
    stride = len(padded_input_signal) // num_splines
    import scipy.signal
    # we construct the spline basis by building the first one, then the rest
    # are identical copies offset by stride
    first_spline = scipy.signal.bspline(numpy.arange(-spline_order, num_splines - spline_order, 1.0 / stride), spline_order)
    first_spline = numpy.roll(first_spline, -spline_order * stride)
    spline_basis = numpy.zeros((num_splines + 1, len(padded_input_signal)), input_signal.dtype)
    for i in range(num_splines + 1):
        spline_basis[i, :] = numpy.roll(first_spline, i * stride)
    spline_basis[:(num_splines // 4), (len(padded_input_signal) // 2):] = 0.0
    spline_basis[(num_splines * 3 // 4):, :(len(padded_input_signal) // 2)] = 0.0
    coefficients = numpy.linalg.lstsq(spline_basis.T, padded_input_signal, rcond=None)
    recon = numpy.dot(coefficients[0], spline_basis)
    start_offset = (len(padded_input_signal) - len(input_signal)) // 2
    return recon[start_offset:(start_offset + len(input_signal))]


def wavelet(input_signal, wavelet_shape, threshold):
    import pywt
    # we have to pad the signal to make it a power of two
    next_power_of_two = int(numpy.floor(numpy.log2(len(input_signal))) + 1)
    padded_input_signal = _pad(input_signal, 2**next_power_of_two)
    wt_coeffs = pywt.wavedec(padded_input_signal, wavelet_shape, level=None, mode='periodization')
    denoised_coeffs = wt_coeffs[:]
    denoised_coeffs[1:] = (pywt.threshold(i, value=threshold) for i in denoised_coeffs[1:])
    recon = pywt.waverec(denoised_coeffs, wavelet_shape, mode='periodization')
    start_offset = (len(padded_input_signal) - len(input_signal)) // 2
    return recon[start_offset:(start_offset + len(input_signal))]
