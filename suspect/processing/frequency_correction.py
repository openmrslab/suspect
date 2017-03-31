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


def spectral_registration(data, target, initial_guess=(0.0, 0.0), frequency_range=None, time_range=None):
    """
    Performs the spectral registration method to calculate the frequency and
    phase shifts between the input data and the reference spectrum target. The
    frequency range and time range over which the two spectra are compared can 
    be specified to exclude regions where the spectra differ, or the signal 
    quality is low. 
    
    Algorithm described in : Near, Jamie, et al. "Frequency and phase drift 
    correction of magnetic resonance spectroscopy data by spectral registration 
    in the time domain." Magnetic resonance in medicine 73.1 (2015): 44-50.
    
    Parameters
    ----------
    data : MRSData
        Input signal to align
        
    target : MRSData
        Input signal data will be aligned to
        
    initial_guess : tuple
        Initial guess for the optimal (frequency,phase) shifts, used as a starting
        point for the scipy.optimize.leastsq function
        
    frequency_range : tuple
        (lower bounds, upper bound) of the frequency range to consider for 
        alignment. Specified in Hz.
    
    time_range : tuple
        (lower bound, upper bound) of the time range to use for alignment. 
    
    Returns
    -------
    freq_and_phase_params : tuple
        Optimal (frequency shift, phase shift) parameters output by the 
        least squares algorithm
    
    """

    # make sure that there are no extra dimensions in the data
    data = data.squeeze()
    target = target.squeeze()

    # the supplied frequency range can be none, in which case we use the whole
    # spectrum, or it can be a tuple defining two frequencies in Hz, in which
    # case we use the spectral points between those two frequencies, or it can
    # be a numpy.array of the same size as the data in which case we simply use
    # that array as the weightings for the comparison

    # Specify some default spectral weights (i.e. equal weighting to all region of the spectrum)
    spectral_weights = numpy.ones(data.shape)
    if type(frequency_range) is tuple:
        spectral_weights = numpy.logical_and(data.frequency_axis() > frequency_range[0],data.frequency_axis() < frequency_range[1])
    elif frequency_range is not None:
        # If frequency_range is a numpy array...
        spectral_weights = frequency_range
    
    # Apply the spectral weights to the data
    d = data.spectrum()*spectral_weights
    t = target.spectrum()*spectral_weights
    
    # Convert back to the time-domain MRSData object
    data = d.fid()
    target = t.fid()
    
    # Check on time_axis constraints
    # the supplied time range can be None, in which case the whole signal will 
    # used, or it can be a tuple specifying the lower and upper bounds of the 
    # time range to use. 
    if type(time_range) is tuple:
        # Run the algorithm on a subset of the time domain samples only, defined by the input range
        time_idx = numpy.logical_and(data.time_axis()>=time_range[0],data.time_axis()<=time_range[1])
        data = data[time_idx]
        target = target[time_idx]
            

    # define a residual function for the optimizer to use
    def residual(input_vector):
        transformed_data = data.adjust_frequency(-input_vector[0]).adjust_phase(-input_vector[1])
        residual_data = transformed_data - target
        return_vector = numpy.zeros(len(residual_data) * 2)
        return_vector[:len(residual_data)] = residual_data.real
        return_vector[len(residual_data):] = residual_data.imag
        return return_vector

    out = scipy.optimize.leastsq(residual, initial_guess)
    freq_and_phase_params = (out[0][0], out[0][1])
    
    return freq_and_phase_params
    
def select_target_for_spectal_registration(data, frequency_range = None):
    """
    Identifies a candidate target signal from an ensemble that can be used for 
    spectral registration. The candidate is the signal whose maximum magnitude in the
    frequency domain is closest to the median maximum abs values of all the 
    signals in the ensemble.
    
    Parameters
    ----------
    data : MRSData
        Ensemble of candidate signals from which the target is selected. 
        
    frequency_range : tuple 
        Specifies the upper and lower bounds in the frequency domain to use for
        selecting the target. Range is specified in Hz.
    
    Returns
    -------
    target_idx : integer 
        Index of the selected target signal
    
    """
    if type(frequency_range) is tuple:
        spectral_weights = numpy.logical_and(data.frequency_axis(),frequency_range[0],data.frequency_axis() < frequency_range[1])
    else:
        spectral_weights = frequency_range
    
    filtered_spectrum = spectral_weights*data.spectrum()
    max_vals = numpy.max(numpy.abs(filtered_spectrum),axis=1)
    target_idx = numpy.argmin(numpy.abs(numpy.median(max_vals)-max_vals))

    return target_idx
    
