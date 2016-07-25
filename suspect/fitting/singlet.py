import lmfit
import numpy
import scipy.optimize
import numbers

import suspect


def complex_to_real(complex_fid):
    """
    Standard optimization routines as used in lmfit require real data. This
    function takes a complex FID and constructs a real version by concatenating
    the imaginary part to the complex part. The imaginary part is also reversed
    to keep the maxima at each end of the FID and avoid discontinuities in the
    center.

    :param complex_fid: the complex FID to be converted to real.
    :return: the real FID, which has twice as many points as the input.
    """
    np = complex_fid.shape[0]
    real_fid = numpy.zeros(np * 2)
    real_fid[:np] = complex_fid.real
    real_fid[np:] = complex_fid.imag[::-1]
    return real_fid


def real_to_complex(real_fid):
    """
    Standard optimization routines as used in lmfit require real data. This
    function takes a real FID generated from the optimization routine and
    converts it back into a true complex form.

    :param real_fid: the real FID to be converted to complex.
    :return: the complex version of the FID
    """
    np = int(real_fid.shape[0] / 2)
    complex_fid = numpy.zeros(np, 'complex')
    complex_fid[:] = real_fid[:np]

    # the imaginary part of the FID has been reversed, have to flip it back
    imag_fid = real_fid[np:]
    complex_fid += 1j * imag_fid[::-1]
    return complex_fid


# metabolite_name_list = []


def phase_fid(fid_in, phase0, phase1):
    """
    This function performs a Fourier Transform on the FID to shift it into phase.

    :param fid_in: FID to be fitted.
    :param phase1: phase1 value.
    :param phase0: phase0 value.
    :return:
    """
    spectrum = numpy.fft.fftshift(numpy.fft.fft(fid_in))
    np = fid_in.np
    phase_shift = phase0 + phase1 * numpy.linspace(-np / 2, np / 2, np, endpoint=False)
    phased_spectrum = spectrum * numpy.exp(1j * phase_shift)
    return fid_in.inherit(numpy.fft.ifft(numpy.fft.ifftshift(phased_spectrum)))


def make_basis(params, time_axis):
    """
    This function generates a basis set.

    :param params: lmfit Parameters object containing fitting parameters.
    :param time_axis: the time axis.
    :return: a matrix containing the generated basis set.
    """
    metabolite_name_list = []
    for param in params.keys():
        split = param.split('_')
        if len(split) == 2:
            if split[0] not in metabolite_name_list:
                metabolite_name_list.append(split[0])

    basis_matrix = numpy.matrix(numpy.zeros((len(metabolite_name_list), len(time_axis) * 2)))
    for i, metabolite_name in enumerate(metabolite_name_list):
        gaussian = suspect.basis.gaussian(time_axis,
                                          params["{}_frequency".format(metabolite_name)],
                                          params["{}_phase".format(metabolite_name)].value,
                                          params["{}_width".format(metabolite_name)])
        real_gaussian = complex_to_real(gaussian)
        basis_matrix[i, :] = real_gaussian
    return basis_matrix


def do_fit(params, time_axis, real_unphased_data):
    """
    This function performs the fitting.

    :param params: lmfit Parameters object containing fitting parameters.
    :param time_axis: the time axis.
    :param real_unphased_data:
    :return: List of fitted data points.
    """
    baseline_points = 16
    basis = make_basis(params, time_axis)

    weights = scipy.optimize.nnls(basis[:, baseline_points:-baseline_points].T,
                                  real_unphased_data[baseline_points:-baseline_points])[0]

    fitted_data = numpy.array(numpy.dot(weights, basis)).squeeze()
    return fitted_data


def residual(params, time_axis, data):
    """
    This function calculates the residual to be minimized by the least squares means method.

    :param params: lmfit Parameters object containing fitting parameters.
    :param time_axis: the time axis.
    :param data: FID to be fitted.
    :return: residual values of baseline points.
    """
    baseline_points = 16
    # unphase the data to make it pure absorptive
    unphased_data = phase_fid(data, -params['phase0'], -params['phase1'])
    real_unphased_data = complex_to_real(unphased_data)

    fitted_data = do_fit(params, time_axis, real_unphased_data)
    res = fitted_data - real_unphased_data

    return res[baseline_points:-baseline_points]


def fit_data(data, initial_params):
    """
    This function takes an FID and a set of parameters contained in an lmfit Parameters object,
    and fits the data using the least squares means method.

    :param data: FID to be fitted.
    :param initial_params: lmfit Parameters object containing fitting parameters.
    :return: tuple of weights as a list, data as a list, and result as an lmift MinimizerResult object.
    """
    baseline_points = 16
    fitting_result = lmfit.minimize(residual,
                                    initial_params,
                                    args=(data.time_axis(), data),
                                    xtol=5e-3)

    unphased_data = phase_fid(data,
                              -fitting_result.params['phase0'],
                              -fitting_result.params['phase1'])
    real_unphased_data = complex_to_real(unphased_data)
    real_fitted_data = do_fit(initial_params, data.time_axis(), real_unphased_data)
    fitted_data = real_to_complex(real_fitted_data)
    fitting_basis = make_basis(fitting_result.params, data.time_axis())
    fitting_weights = scipy.optimize.nnls(fitting_basis[:, baseline_points:-baseline_points].T,
                                          real_unphased_data[baseline_points:-baseline_points])[0]

    return fitting_weights, fitted_data, fitting_result
