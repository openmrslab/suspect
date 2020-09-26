import lmfit
import numpy as np
import json
import functools
import operator

import suspect.basis


# this is the underlying function for the GaussianPeak model class
def gaussian(in_data, amplitude, frequency, phase, fwhm):
    return amplitude * in_data.inherit(suspect.basis.gaussian(in_data.time_axis(), frequency, phase, fwhm))


# this is the underlying function for the GlobalPhase model class
def phase_shift(in_data, phase0, phase1):
    return in_data.spectrum().inherit(np.ones_like(in_data)).adjust_phase(phase0, phase1)


# this is the underlying function for combining the models together
def apply_in_freq_domain(model, phase_shift):
    return (model.spectrum() * phase_shift).fid()


class GaussianPeak(lmfit.Model):
    """
    Class to represent a Gaussian peak for fitting.

    The Gaussian peak is parameterised by 4 values: amplitude, frequency,
    phase, and FWHM (full width at half maximum).

    Each parameter can be specified in three different ways:
    1. Passing a numeric value sets the initial guess for that parameter
    2. Passing a string of a number fixes that parameter to that value
    3. Passing a dictionary allows setting any of the constraints supported
       by the underlying LMFit parameter: value, min, max, vary, and expr

    By default the phase of the peak will be fixed at 0 and the amplitude
    and FWHM will be constrained to be bigger than 0 and 1Hz respectively.

    Parameters
    ----------
    name
        The name of the peak
    amplitude
        The amplitude (area) of the peak
    frequency
        The frequency of the peak in Hertz
    phase
        The phase of the peak in radians
    fwhm
        The full width at half maximum of the peak in Hertz
    """

    def __init__(self, name, amplitude=1, frequency=0, phase="0", fwhm=20):
        lcls = locals()
        params = {p: lcls[p] for p in ["amplitude", "phase", "frequency", "fwhm"]}

        super().__init__(gaussian, prefix="{}_".format(name))

        for name, value in params.items():
            if isinstance(value, str):
                self.set_param_hint(name, value=float(value), vary=False)
            elif isinstance(value, dict):
                self.set_param_hint(name, **value)
            else:
                self.set_param_hint(name, value=value)
        if not "min" in self.param_hints["amplitude"]:
            self.set_param_hint("amplitude", min=0)
        if not "min" in self.param_hints["fwhm"]:
            self.set_param_hint("fwhm", min=1)


class Model:
    """
    A model of an MRS FID signal which can be fitted to data.

    This model is created by passing a set
    of individual peak models, to which it then appends a phase model. By
    default the first order phase is constrained to be 0.

    Parameters
    ----------
    peak_models
        The descriptions of the peaks making up the model.
    phase0
        The estimated zero order phase in radians.
    phase1
        The estimated first order phase in radians per Hz.
    """
    def __init__(self, peak_models, phase0=0, phase1="0"):
        phase_model = lmfit.model.Model(phase_shift)
        phase_model.set_param_hint("phase0", value=0)
        phase_model.set_param_hint("phase1", value=0, min=0, max=16e-3)

        params = {
            "phase0": phase0,
            "phase1": phase1
        }
        for name, value in params.items():
            if isinstance(value, str):
                phase_model.set_param_hint(name, value=float(value), vary=False)
            elif isinstance(value, dict):
                phase_model.set_param_hint(name, **value)
            else:
                phase_model.set_param_hint(name, value=value)

        self.composite_model = lmfit.model.CompositeModel(peak_models,
                                                          phase_model,
                                                          apply_in_freq_domain)

    def fit(self, data, baseline_points=4):
        """
        Perform a fit of the model to an FID.

        Parameters
        ----------
        data
            The time domain data to be fitted.
        baseline_points
            The first baseline_points of the FID will be ignored in the fit.

        Returns
        -------
            ModelResult
        """
        params = self.composite_model.make_params()
        weights = np.ones_like(data, dtype=np.float)
        weights[:baseline_points] = 0
        return self.composite_model.fit(data,
                                        params=params,
                                        in_data=data,
                                        weights=weights)

    @classmethod
    def load(cls, filename):
        with open(filename) as fin:
            model_dict = json.load(fin)
        return cls.from_dict(model_dict)

    @classmethod
    def from_dict(cls, model_dict):
        """
        Create a model from a dict.

        Parameters
        ----------
        model_dict
            dict describing the model.

        Returns
        -------
        Model
            The specified model ready for fitting.
        """
        phase0 = model_dict.get("phase0", 0)
        phase1 = model_dict.get("phase1", "0")

        peak_params = (GaussianPeak(k, **v) for (k, v) in model_dict.items()
                       if k not in ["phase0", "phase1"])
        peak_models = functools.reduce(operator.add, peak_params)

        return cls(peak_models, phase0, phase1)


