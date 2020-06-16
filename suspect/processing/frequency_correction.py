import numpy as np
import scipy.optimize

import suspect


def residual_water_alignment(data):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """

    # get rid of any extraneous dimensions to the data
    data = data.squeeze()
    current_spectrum = np.fft.fft(data)
    peak_index = np.argmax(np.abs(current_spectrum))
    if peak_index > len(data) / 2:
        peak_index -= len(data)
    return peak_index * data.df


def spectral_registration(data, target, initial_guess=(0.0, 0.0), frequency_range=None, **kwargs):
    """
    Performs the spectral registration method [2]_ to calculate the frequency and
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
    frequency_shift : float
        The estimated frequency shift in Hz.
    phase_shift : float
        The estimated phase shift in radians.

    References
    ----------
    .. [2] Near, J., Edden, R., Evans, C. J., Paquin, R., Harris, A., & Jezzard, P. (2014). Frequency and phase drift correction of magnetic resonance spectroscopy data by spectral registration in the time domain. Magnetic Resonance in Medicine, 73(1), 44–50. http://doi.org/10.1002/mrm.25094
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
        spectral_weights = np.logical_and(frequency_range[0] < data.frequency_axis(),
                                          frequency_range[1] > data.frequency_axis())
    elif type(frequency_range) is slice:
        spectral_weights = np.zeros_like(target, np.bool)
        spectral_weights[frequency_range] = 1
    else:
        spectral_weights = frequency_range

    # define a residual function for the optimizer to use
    def residual(input_vector):
        transformed_data = data.adjust_frequency(-input_vector[0]).adjust_phase(-input_vector[1])
        residual_data = transformed_data - target
        if frequency_range is not None:
            spectrum = residual_data.spectrum()
            weighted_spectrum = spectrum * spectral_weights
            # remove zero-elements
            weighted_spectrum = weighted_spectrum[weighted_spectrum != 0]
            residual_data = np.fft.ifft(np.fft.ifftshift(weighted_spectrum))
        return_vector = np.zeros(len(residual_data) * 2)
        return_vector[:len(residual_data)] = residual_data.real
        return_vector[len(residual_data):] = residual_data.imag
        return return_vector

    out = scipy.optimize.leastsq(residual, initial_guess)
    return out[0][0], out[0][1]


def rats(data, target, initial_guess=(0.0, 0.0), frequency_range=None, baseline_order=2, **kwargs):
    """
    Uses the RATS (Robust Alignment to a Target Spectrum) [1]_ to calculate the
    frequency and phase shifts between the input data and a reference
    spectrum. RATS operates in the frequency domain and the frequencies
    used to align the spectra can be specified to exclude regions where the
    spectra differ.

    Parameters
    ----------
    data : MRSData
        The data to be aligned to the target
    target : MRSData
        The target data to which the moving data will be aligned
    initial_guess : tuple
        A 2-tuple of frequency and phase shifts at which the optimisation
        routine will start searching. See below for more information.
    frequency_range : tuple, slice or ndarray
        The frequency range can be specified in multiple different ways: a
        2-tuple containing low and high frequency cut-offs in Hertz for the
        comparison, as a slice object into the spectrum (for use with the
        slice_ppm() function, or as an array of weights to apply to the
        spectrum.
    baseline_order : int
        The order of the polynomial baseline.

    Returns
    -------
    frequency_shift : float
        The estimated frequency shift in Hz.
    phase_shift : float
        The estimated phase shift in radians.

    Notes
    -----
    Experimentally, I have found the RATS method to sometimes get stuck in
    local minima where the frequency shift is larger than about 6Hz. Although
    this can be addressed via the `initial_guess` parameter, when batch
    processing a sequence of spectra experiencing substantial drift, there is
    no single good value of `initial_guess`. In order to address this issue,
    this function begins by using a coarse grid based search to evaluate
    frequencies every 2Hz from 20Hz above the `initial_guess` frequency to 20Hz
    below. It then picks from those the frequency which gives best alignment
    with the target data as a starting point for the optimisation search. So
    far this has been very effective, but in case of discovering any problems,
    please raise an issue at http://github.com/openmrslab/suspect.

    References
    ----------
    .. [1] Wilson, M. (2018). Robust retrospective frequency and phase correction for single-voxel MR spectroscopy. Magnetic Resonance in Medicine, 81(5), 2878–2886. http://doi.org/10.1002/mrm.27605
    """

    if type(frequency_range) is tuple:
        included_frequencies = np.logical_and(frequency_range[0] < data.frequency_axis(),
                                              frequency_range[1] > data.frequency_axis())
    elif type(frequency_range) is slice:
        included_frequencies = np.zeros_like(target, np.bool)
        included_frequencies[frequency_range] = 1
    elif frequency_range is None:
        included_frequencies = np.ones_like(target, np.bool)
    else:
        included_frequencies = frequency_range

    spectral_points = np.count_nonzero(included_frequencies)
    included_target = target.spectrum()[included_frequencies]

    # the VARPRO basis consists of the moving data shifted by the current
    # frequency estimate and a set of polynomials making up the baseline
    # note that the polynomials require order + 1 terms because there is a
    # zero order term as well, so with the spectral data we have order + 2
    # TODO using polynomials for the baseline seems like the wrong choice
    # first row is a placeholder for the spectrum, second is the zero order
    basis = np.ones((spectral_points, baseline_order + 2), 'complex')
    if baseline_order > 0:
        # linear term
        basis[:, 2] = np.arange(-spectral_points // 2, spectral_points // 2)
        # all higher order terms are just raising the linear term to higher powers
        for i in range(2, baseline_order + 1):
            basis[:, i + 1] = np.power(basis[:, 2], i)

    def optimise_phase(frequency_corrected_data):
        basis[:, 0] = frequency_corrected_data
        weights = np.linalg.lstsq(basis, included_target)[0]
        return np.angle(weights[0])

    # define our cost function closure
    def cost(frequency):
        frequency_corrected_data = data.adjust_frequency(frequency).spectrum()[included_frequencies]
        phase_shift = optimise_phase(frequency_corrected_data)
        return np.linalg.norm(included_target - frequency_corrected_data.adjust_phase(phase_shift))

    # it turns out that finding a good bracket for Brent is not trivial if the shift is
    # larger than about 7 Hz. therefore, we start with a coarse grid search to find the
    # smallest value within +/-20Hz and start there
    frequency_grid = np.linspace(initial_guess[0] - 20, initial_guess[0] + 20, 20)
    cost_grid = np.zeros_like(frequency_grid)
    for i in range(20):
        cost_grid[i] = cost(frequency_grid[i])
    lowest_frequency = frequency_grid[np.argmin(cost_grid)]

    frequency_correction = scipy.optimize.brent(cost, brack=(lowest_frequency - 10,
                                                             lowest_frequency,
                                                             lowest_frequency + 10))
    fc_data = data.adjust_frequency(frequency_correction).spectrum()[included_frequencies]
    phase_correction = optimise_phase(fc_data)
    return -frequency_correction, -phase_correction


def correct_frequency_and_phase(data, target, method='sr', axis=-1, **kwargs):
    """
    Interface to frequency and phase correction algorithms, but returning the
    corrected data rather than the calculated shifts. Can be applied to
    multi-dimensional data to align e.g. a sequence of spectra at once.

    Parameters
    ----------
    data : MRSBase
        The data to be corrected.
    target : MRSBase
        The reference spectrum to which data will be aligned.
    method : str, optional
        The correction method to be used. Should be one of:

            - 'sr'      Time-domain Spectral Registration - see :meth:`spectral_registration`
            - 'rats'    RATS (Robust Alignment to a Target Spectrum) - see :meth:`rats`
            - 'rwa'     Residual Water Alignment - see :meth:`residual_water_alignment`
            - custom    A callable object, see below

    axis : int or None, optional
        The axis defining the spectral dimension.
    kwargs
        Arguments to be passed through to the method function.

    Returns
    -------
    MRSBase
        The data with corrected frequency and phase.

    Notes
    -----
    **Custom Frequency/Phase Corrections**

    If you have an alternative method for calculating frequency and phase
    shifts then you can simply pass a callable to the ``method`` parameter.

    The callable is called as ``method(moving_data, target_data, **kwargs)``
    where `kwargs``corresponds to any other parameters that may be passed in
    such as `initial_guess`. The method shall return a 2-tuple of floats
    containing the frequency shift in Hertz and the phase shift in radians
    between the first passed data and the second. Note that it is the
    measured frequency and phase shifts that should be returned, they will
    be negated by this function to correct the spectrum to the target.
    """
    if method == 'sr':
        func = spectral_registration
    elif method == 'rats':
        func = rats
    elif method == 'rwa':
        func = residual_water_alignment
    elif callable(method):
        func = method
    else:
        raise ValueError("Unknown correction method {0}".format(method))

    # define a closure function to calculate shifts and perform the alignment
    def correct(moving_data: suspect.MRSBase):
        frequency_shift, phase_shift = func(moving_data, target, **kwargs)
        return moving_data.adjust_frequency(-frequency_shift).adjust_phase(-phase_shift)

    if len(data.shape) == 1:
        return correct(data)
    else:
        return np.apply_along_axis(correct, axis, data)
