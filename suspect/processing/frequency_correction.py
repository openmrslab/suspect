import numpy as np
import scipy.optimize



def peak_alignment(x_axis, data, window, ref_idx=0, max_type='abs', pad_with = 'replicate'):
    """
    Performs a simple shift alignment of an ensemble of signals, using the maximum values of each signal within a specified window as a reference point. These peak values are all aligned to the corresponding peak value in the signal specified by ref_idx.
    :param x_axis: axis along which the signals are aligned. len(x_axis) = n_points
    :param data: Ensemble of signals to be aligned. Dimensions assumed to be [n_signals x n_points].
    :param window: list [lower_bound, upper_bound] specifying the range of values in x_axis to use for identification of the peak value in each signal
    :param ref_idx: Index of the row in the data matrix to align all of the other signals to. Default is to use the first one.
    :param max_type: optional, default = 'abs'. If data is complex, defines the component of the data to use for determining peak values. Other options, if data is complex: ['real','imag']
    :param pad_with: Optional, default = 'replicate'. Defines the type of data to use to pad the ends of the signals if shifting is required to align the ensemble. Valid options are ['replicate','zero','circular']. 'replicate' pads beginning/end of the signal with replicas of the first/last data points. 'zero' pads with zeros, 'circular' performs a circular shift.
    :return: 
    aligned_data: Array of aligned signals, the same size as the input 'data' array.
    peak_deltas: Array of values corresponding to the amount of shift applied to each of the signals in the ensemble to align it with the reference. The values are in the units of the input x_axis
    """
    
    # Get the indices of the x_axis that fall within the specified window
    x_axis = np.array(x_axis)
    ix = np.where(np.all([x_axis>=np.min(window),x_axis<=np.max(window)],axis=0))[0]

    d_subset = data[:,ix]
    if max_type is 'abs':
        d_subset = np.abs(d_subset)
    elif np.iscomplexobj(data): 
        if max_type is 'real':
            d_subset = np.real(d_subset)
        elif max_type is 'imag':
            d_subset = np.imag(d_subset)
    elif max_type is 'imag':
        # This doesn't make sense...
        print('data is not a complex array, so max_type can''t be ''imag''')
        return
    
    # Get the indices of the largest values in this subset of the data
    m_ix = np.argmax(d_subset,axis=1)

    # Get the peak index of the reference signal
    ref_peak_idx = m_ix[ref_idx]
    
    #Align all the signals to the reference signal
    # Make an array to hold the output data
    if np.iscomplexobj(data):
        aligned_data = np.zeros(data.shape,dtype=np.complex)
    else:
        aligned_data = np.zeros(data.shape)
    
    def modify_shift(x,pad_with,shift_val):
        
        # Apply custom padding parameters
        if shift_val > 0:
            if pad_with is 'zero':
                x[-shift_val:]=0
            elif pad_with is 'replicate':
                x[-shift_val:]=x[-shift_val]
        else:
            if pad_with is 'zero':
                x[0:-shift_val]=0
            elif pad_with is 'replicate':
                x[0:-shift_val]=x[-shift_val]
        return x
    
    for ii in range(data.shape[0]):
        shift_val = m_ix[ii]-ref_peak_idx
        if shift_val !=0:
            # Perform a circular shift
            shifted = np.roll(data[ii,:],-shift_val)
            # Check 
            if pad_with is not 'circular':
                shifted = modify_shift(shifted,pad_with,shift_val)

        else:
            shifted = data[ii,:]
        
        aligned_data[ii,:] = shifted
    
    # Get the delta values (in the units of the x_axis) each signal had to be shifted to align with the reference signal
    peak_deltas = x_axis[ix[m_ix]] - x_axis[ix[m_ix[ref_idx]]]
    
    return aligned_data, peak_deltas



def spectral_registration(data, target = 'auto_select', initial_guess=(0.0, 0.0), ppm_range=None):
    """
    Performs the spectral registration method to calculate the frequency and
    phase shifts between the input data and the reference spectrum target. The
    frequency range over which the two spectra are compared can be specified to
    exclude regions where the spectra differ.

    :param data: MRS data object
    :param target: Description of the signal to use as a target. Default = 'auto_select', which automatically selects a signal from the input array that is closest to the median of all the signals in the array. This method is recommended if there is a possibility that some of the scans might be outliers. Other valid inputs: The actual reference signal to use, or the index of the signals to use as the reference from the data array.
    :param initial_guess:
    :param ppm_range:
    :return:
    """

    # make sure that there are no extra dimensions in the data
    data = data.squeeze()
    
    # Check on the logic of the inputs a bit
    if data.ndim > 2:
        print('Improper size of the ''data'' array. The array can have at most 2 dimensions.')
        return
    elif data.ndim == 1 and target is 'auto_select':
        # Nothing to align to!
        print('There must be at least two signals in the ''data'' array to performan alignment.')
        return
    elif (data.shape[0] == 1 or data.shape[1]==1) and target is 'auto_select':
        # Weird edge case of a 1D array that looks like a 2D array when you check ndims
        print('Improper size of the ''data'' array. The array must have at least 2 dimensions.')
        return

    
    # The supplied ppm range can be none, in which case we use the whole
    # spectrum, or it can be a tuple defining upper and lower bounds, in which
    # case we use the spectral points between those two values, or it can
    # be a numpy.array of the same size as the data in which case we simply use
    # that array as the weightings for the comparison
    if ppm_range is not None:
        if type(ppm_range) is tuple:
            
            spectral_weights = np.array((data.frequency_axis_ppm() >= ppm_range[0]) & (data.frequency_axis_ppm() <= ppm_range[1]),dtype=int)
            
        else:
            spectral_weights = np.array(ppm_range)
    else:
        spectral_weights = np.ones(data.shape[1],dtype=bool) 
    
    # Identify target 
    if target is 'auto_select':
        # Use the autoselection method to identify a suitable target from the input array
        if data.shape[0]==2:
            target = data[0,:]
            # Print something about how there were only 2 signals, so we just selected the first one
        else:
            # Find a reference signal to use. Use only the 
            target_idx = suggest_target(data * spectral_weights)
            target = data[target_idx,:]
            
    elif len(target)==1:
        # Assume this is an index to use
        target = data[target,:]
        
    # Define the residual function
    def residual(p, x, y):

#        p = Parameters to apply     
#        x = Signal to adjust
#        y = Target
        
        # Apply frequency and phase shift
        transformed_data = transform_fid(x, p[0], p[1])
        
        # Compute the elementwise deltas between the signal and the target
        residual_data = transformed_data - y
        
        spectrum = residual_data.spectrum() # Frequency domain representation of the data
        
        # Apply spectral weights (if there are any)
        weighted_spectrum = spectrum * spectral_weights

        # Convert back to time domain
        residual_data = np.fft.ifft(np.fft.ifftshift(weighted_spectrum))
            
        # Concatenate the real and imaginary components so the optimization fcn can compute the mse
        return_vector = np.zeros(len(residual_data) * 2)
        return_vector[:len(residual_data)] = residual_data.real
        return_vector[len(residual_data):] = residual_data.imag
        return return_vector

    # Align all the signals in the input array to the target
    # Create an array to hold the aligned data
    aligned_fids = np.zeros(data.shape,dtype=np.complex)
    opt_params = []
    for ii in range(data.shape[0]):
        # Run the optimization on a slice of data
        out = scipy.optimize.leastsq(residual, initial_guess, args= (data[ii,:],target), maxfev=1000 )
        # Apply the parameters to the original signal 
        aligned_fids[ii,:] = transform_fid(data[ii,:], out[0][0], -out[0][1])
        opt_params = opt_params + [out[0][0], -out[0][1]]
    
    aligned_data = data.inherit(aligned_fids)
    
    return aligned_data, target, opt_params


def transform_fid(fid, frequency_shift, phase_shift):
    # Apply a frequency and phase shift to an FID signal
    time_axis = fid.time_axis()
    correction = np.exp(2j * np.pi * (frequency_shift * time_axis + phase_shift))
    transformed_fid = np.multiply(fid, correction)
    return transformed_fid
    
def suggest_target(data):
    
    if np.iscomplexobj(data):
        data = np.abs(data)
        
    # Compute the median of the signals in the array     
    m = np.median(data,axis=0)
    
    # Compute the sum of the magnitude of the distance between the median of the signals and each individual signal 
    
    # The target should be the one that minimizes this total
    target_idx = np.argmin( np.sum( np.abs( m-data ),axis=1 ) )
    return target_idx
