import numpy
import scipy.optimize


def residual_water_alignment(data):

    # get rid of any extraneous dimensions to the data
    data = data.squeeze()
    current_spectrum = numpy.fft.fft(data)
    peak_index = numpy.argmax(numpy.abs(current_spectrum))
    if peak_index > len(data) / 2:
        peak_index -= len(data)
    return peak_index * data.df


def spectral_registration(data, target, initial_guess=(0.0, 0.0), frequency_range=None):
    """
    Performs the spectral registration method to calculate the frequency and
    phase shifts between the input data and the reference spectrum target. The
    frequency range over which the two spectra are compared can be specified to
    exclude regions where the spectra differ.

    Parameters
    ----------
    data : MRSData
    target : MRSData
    initial_guess : tuple
    frequency_range : tuple, slice or ndarray
        The frequency range can be specified in multiple different ways: a
        2-tuple containing low and high frequency cut-offs in Hertz for the
        comparison, as a slice object into the spectrum (for use with the
        slice_ppm() function, or as an array of weights to apply to the
        spectrum.

    Returns
    -------

    """

    # make sure that there are no extra dimensions in the data
    data = data.squeeze()
    target = target.squeeze()

    # the supplied frequency range can be none, in which case we use the whole
    # spectrum, or it can be a tuple defining two frequencies in Hz, in which
    # case we use the spectral points between those two frequencies, or it can
    # be a numpy.array of the same size as the data in which case we simply use
    # that array as the weightings for the comparison
    if type(frequency_range) is tuple:
        spectral_weights = numpy.logical_and(frequency_range[0] < data.frequency_axis(),
                                             frequency_range[1] > data.frequency_axis())
    elif type(frequency_range) is slice:
        spectral_weights = numpy.zeros_like(target, numpy.bool)
        spectral_weights[frequency_range] = 1
    else:
        spectral_weights = frequency_range

    # define a residual function for the optimizer to use
    def residual(input_vector):
        #transformed_data = transform_fid(data, -input_vector[0], -input_vector[1])
        transformed_data = data.adjust_frequency(-input_vector[0]).adjust_phase(-input_vector[1])
        residual_data = transformed_data - target
        if frequency_range is not None:
            spectrum = residual_data.spectrum()
            weighted_spectrum = spectrum * spectral_weights
            # remove zero-elements
            weighted_spectrum = weighted_spectrum[weighted_spectrum != 0]
            residual_data = numpy.fft.ifft(numpy.fft.ifftshift(weighted_spectrum))
        return_vector = numpy.zeros(len(residual_data) * 2)
        return_vector[:len(residual_data)] = residual_data.real
        return_vector[len(residual_data):] = residual_data.imag
        return return_vector

    out = scipy.optimize.leastsq(residual, initial_guess)
    return out[0][0], out[0][1]
