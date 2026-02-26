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
    # normalize so they sum to 1
    channel_weights = channel_weights/numpy.sum(numpy.abs(channel_weights))

    # try some basic phase correction
    # in our truncation of the SVD to rank 1, we know that v[0] is our FID
    # use the first point of it to phase the signal
    phase_shift = numpy.angle(v[0, 0])
    
    channel_weights_phase_corrected = channel_weights * numpy.exp(-1j * phase_shift)        

    return channel_weights_phase_corrected, channel_weights, phase_shift


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
    

def RAMRS_SVD_weighting(data, phase_correct = True, whiten = False, W = None, noise_range = None):
    """
    This channel combination approach operates on raw MRS (spectral) data, 
    prior to averaging repetitions from multiple acquisitions. It is useful in 
    cases where there is instability in frequency and phase shifts across 
    averages. Coil weights are determined from the entire 3D array, and applied
    to each average. Frequency and phase correction can then be performed on 
    the channel-combined acquisitions, prior to final averaging. 
    
    This channel combination method is described in the 'Multiple Acquisitions' 
    section of:

    Rodgers, Christopher T., and Matthew D. Robson. "Receive array magnetic 
    resonance spectroscopy: whitened singular value decomposition (WSVD) gives
    optimal Bayesian solution." Magnetic resonance in medicine 63.4 
    (2010): 881-891.
    
    Performing signal whitening, prior to channel combination, is optional. You 
    can also (optionally) pass in a whitening matrix, W, if it has already been
    computed, or derive one from the data, as described in the reference.
    
    Parameters
    ----------
    data : MRSSpectrum
        Input spectrum matrix for channel combination. The data should be a 3D 
        array, with dimensions [num_acquisitions, n_channels, n_samples], and
    
    phase_correct : bool
        If True, perform a 0th order phase correction along with the channel
        weighting
    
    whiten : bool
        (Optional) If True, whiten the array prior to channel combination
    
    W : array
        (Optional) Whitening array to use. If W = None and whiten = True, the
        array will be computed from the data
    
    noise_range : tuple
        (Optional) (lower bound, upper bound) of the frequency range to use for
        deriving the whitening matrix. Specified in Hz.
    
    Returns
    -------
    combined_data : MRSSpectrum
        Output spectrum, after coil combination.
    
    chann_weights_phase_corrected : array
        Channel weights derived from the data, including the 0th order phase
        correction term. If phase_correct = True, these are the weights that
        were applied to the data
    
    channel_weights : array
        Weights applied to each channel, normalized so they sum to 1. If 
        phase_correct = False, these are the weights that were applied to the 
        averages.
    
    phase_shift : float
        0th order phase correction term derived from the SVD. 
    
    W : array
        Whitening matrix used, or None if pre-whitening was not done.       
    
    """
    
    n_reps, n_chans, n_samps = data.shape
    
    # Re-structure the data to get it in the right format
    R = numpy.transpose(data,(0,2,1))
    d = R.shape
    R = numpy.reshape(R,(d[0]*d[1],d[2]))
    
    if whiten:
        if W is None:
            # Derive the whitening matrix from this data
            # Isolate a subset of the spectrum that is in the noise region
            noise_idx = numpy.logical_and(data.frequency_axis()>=noise_range[0], data.frequency_axis()<=noise_range[1])
            N = data[:,:,noise_idx]
            # Do some reshaping of the array to make it 2D
            N = numpy.transpose(N,(0,2,1))
            d = N.shape
            N = numpy.reshape(N,(d[0]*d[1],d[2]))            
            W = get_whitening_matrix(N)
        
        # Apply whitening matrix to the ensemble        
        S = numpy.dot(W,R.T) 
               
    else:
        # Do not pre-whiten, run SVD on the raw data
        S = R.T
    
    chann_weights_phase_corrected, channel_weights, phase_shift = svd_weighting(S)
    
    # Apply the weights
    if phase_correct:
        Q = combine_channels(S, weights=chann_weights_phase_corrected)
    else:
        Q = combine_channels(S, weights=channel_weights)
    
    combined_spectrum = numpy.reshape(Q,(n_reps,n_samps))
    combined_data = data.inherit(combined_spectrum)

    
    return combined_data, chann_weights_phase_corrected, channel_weights, phase_shift, W
    
    

def get_whitening_matrix(data):
    
    C = numpy.cov(data,rowvar=False)
    
    [w,v]=numpy.linalg.eig(C) # Eigenvalue decomp of covariance matrix
    D = numpy.diag(2*(w**(-0.5)))
    
    # Scaling matrix
    W = numpy.dot(v,D)
    return W



