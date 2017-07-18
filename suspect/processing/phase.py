import lmfit
import numpy as np


def mag_real(data, *args, range_hz=None, range_ppm=None):
    """
    Estimates the zero and first order phase parameters which minimise the
    difference between the real part of the spectrum and the magnitude. Note
    that these are the phase correction terms, designed to be used directly
    in the adjust_phase() function without negation.
    
    Parameters
    ----------
    data: MRSBase
        The data to be phased
    range_hz: tuple (low, high)
        The frequency range in Hertz over which to compare the spectra
    range_ppm: tuple (low, high)
        The frequency range in PPM over which to compare the spectra. range_hz
        and range_ppm cannot both be defined.
    Returns
    -------
    phi0 : float
        The estimated zero order phase correction
    phi1 : float
        The estimated first order phase correction
    """
    if range_hz is not None and range_ppm is not None:
        raise KeyError("Cannot specify both range_hz and range_ppm")

    if range_hz is not None:
        frequency_slice = data.slice_hz(*range_hz)
    elif range_hz is not None:
        frequency_slice = data.slice_ppm(*range_ppm)
    else:
        frequency_slice = slice(0, data.np)

    def single_spectrum_version(spectrum):
        def residual(pars):
            par_vals = pars.valuesdict()
            phased_data = spectrum.adjust_phase(par_vals['phi0'],
                                                par_vals['phi1'])

            diff = np.real(phased_data) - np.abs(spectrum)

            return diff[frequency_slice]

        params = lmfit.Parameters()
        params.add('phi0', value=0, min=-np.pi, max=np.pi)
        params.add('phi1', value=0.0, min=-0.01, max=0.25)

        result = lmfit.minimize(residual, params)
        return result.params['phi0'].value, result.params['phi1'].value

    return np.apply_along_axis(single_spectrum_version,
                               axis=-1,
                               arr=data.spectrum())


def ernst(data):
    """
    Estimates the zero and first order phase using the ACME algorithm, which
    minimises the integral of the imaginary part of the spectrum. Note that
    these are the phase correction terms, designed to be used directly in the
    adjust_phase() function without negation.

    Parameters
    ----------
    data: MRSBase
        The data to be phased
    range_hz: tuple (low, high)
        The frequency range in Hertz over which to compare the spectra
    range_ppm: tuple (low, high)
        The frequency range in PPM over which to compare the spectra. range_hz
        and range_ppm cannot both be defined.
    Returns
    -------
    phi0 : float
        The estimated zero order phase correction
    phi1 : float
        The estimated first order phase correction
    """
    def residual(pars):
        par_vals = pars.valuesdict()
        phased_data = data.adjust_phase(par_vals['phi0'],
                                        par_vals['phi1'])
        return np.sum(phased_data.spectrum().imag)

    params = lmfit.Parameters()
    params.add('phi0', value=0, min=-np.pi, max=np.pi)
    params.add('phi1', value=0.0, min=-0.005, max=0.1)

    result = lmfit.minimize(residual, params, method='simplex')
    return result.params['phi0'].value, result.params['phi1'].value


def acme(data, *args, range_hz=None, range_ppm=None, gamma=100):
    """
    Estimates the zero and first order phase using the ACME algorithm, which
    minimises the entropy of the real part of the spectrum. Note that these
    are the phase correction terms, designed to be used directly in the
    adjust_phase() function without negation.
    
    Parameters
    ----------
    data : MRSBase
        The data to be phased
    range_hz : tuple (low, high)
        The frequency range in Hertz over which to compare the spectra
    range_ppm : tuple (low, high)
        The frequency range in PPM over which to compare the spectra. range_hz
        and range_ppm cannot both be defined.
    gamma : float
        Weighting factor for penalty function.
    Returns
    -------
    phi0 : float
        The estimated zero order phase correction
    phi1 : float
        The estimated first order phase correction
    """
    if range_hz is not None and range_ppm is not None:
        raise KeyError("Cannot specify both range_hz and range_ppm")

    if range_hz is not None:
        frequency_slice = data.slice_hz(*range_hz)
    elif range_hz is not None:
        frequency_slice = data.slice_ppm(*range_ppm)
    else:
        frequency_slice = slice(0, data.np)

    def single_spectrum_version(spectrum):
        def residual(pars):
            par_vals = pars.valuesdict()
            phased_data = spectrum.adjust_phase(par_vals['phi0'],
                                                par_vals['phi1'])

            r = phased_data.real[frequency_slice]
            r = r / np.sum(r)
            derivative = np.abs((r[1:] - r[:-1]))
            derivative_norm = derivative / np.sum(derivative)

            # make sure the entropy doesn't blow up by removing 0 values
            derivative_norm[derivative_norm == 0] = 1

            entropy = -np.sum(derivative_norm * np.log(derivative_norm))

            # penalty function
            p = np.sum(r[r < 0] ** 2)

            return entropy + gamma * p

        params = lmfit.Parameters()
        params.add('phi0', value=0.0, min=-np.pi, max=np.pi)
        params.add('phi1', value=0.001, min=-0.005, max=0.25)

        result = lmfit.minimize(residual, params, method='simplex')
        return result.params['phi0'].value, result.params['phi1'].value

    return np.apply_along_axis(single_spectrum_version,
                               -1,
                               data.spectrum())
