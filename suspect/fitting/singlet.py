import lmfit
import numpy
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
    np = real_fid.shape[0] / 2
    complex_fid = numpy.zeros(np, 'complex')
    complex_fid[:] = real_fid[:np]

    # the imaginary part of the FID has been reversed, have to flip it back
    imag_fid = real_fid[np:]
    complex_fid += 1j * imag_fid[::-1]
    return complex_fid


def gaussian_fid(x, amplitude=1, frequency=0.0, phase=0.0, fwhm=1.0):
    """
    Generates a Gaussian FID for use with the lmfit GaussianFidModel class. The
    helper function complex_to_real is used to convert the FID to a real form.

    :param x: the time axis for the fid data
    :param amplitude: the amplitude of the Gaussian
    :param frequency: the frequency of the Gaussian in Hz
    :param phase: the phase in radians
    :param fwhm: the full width at half maximum of the Gaussian
    :return: a real FID describing a Gaussian relaxation
    """

    complex_fid = amplitude * suspect.basis.gaussian(x, frequency, phase, fwhm)
    return complex_to_real(complex_fid)


class GaussianFidModel(lmfit.models.Model):
    def __init__(self, *args, **kwargs):
        super(GaussianFidModel, self).__init__(gaussian_fid, *args, **kwargs)
        self.set_param_hint('fwhm', min=0)
        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('phase', min=-numpy.pi, max=numpy.pi)

    def guess(self, data=None, **kwargs):
        return self.make_params()

    def copy(self, **kwargs):
        raise NotImplementedError


class Model:
    def __init__(self, peaks):
        self.model = None
        self.params = lmfit.Parameters()
        for peak_name, peak_params in peaks.items():

            current_peak_model = GaussianFidModel(prefix="{}".format(peak_name))
            self.params.update(current_peak_model.make_params())

            for param_name, param_data in peak_params.items():
                # the
                full_name = "{0}{1}".format(peak_name, param_name)

                if full_name in self.params:
                    if type(param_data) is str:
                        self.params[full_name].set(expr=param_data)
                    elif type(param_data) is dict:
                        if "value" in param_data:
                            self.params[full_name].set(value=param_data["value"])
                        if "min" in param_data:
                            self.params[full_name].set(min=param_data["min"])
                        if "max" in param_data:
                            self.params[full_name].set(max=param_data["max"])
                        if "expr" in param_data:
                            self.params[full_name].set(expr=param_data)
                    elif type(param_data) is numbers.Number:
                        self.params[full_name].set(param_data)

            if self.model is None:
                self.model = current_peak_model
            else:
                self.model += current_peak_model

    def fit(self, data):
        fit_result = self.model.fit(complex_to_real(data), x=data.time_axis(), params=self.params)
        result_params = {}
        for component in self.model.components:
            component_name = component.prefix
            result_params[component.prefix] = {}
            for param in component.make_params():
                param_name = str(param).replace(component_name, "")
                result_params[component.prefix][param_name] = fit_result.params[param].value
        fit_curve = real_to_complex(fit_result.best_fit)
        fit_components = {k: real_to_complex(v) for k, v in fit_result.eval_components().items()}
        return {
            "params": result_params,
            "fit": fit_curve,
            "fit_components": fit_components
        }
