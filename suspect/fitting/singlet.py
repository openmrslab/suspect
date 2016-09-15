import lmfit
import numpy
import scipy.optimize
import numbers

import suspect.basis

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


def fit(fid, model, baseline_points=16):
    """
    Fit fid with model parameters.

    :param fid: MRSData object of FID to be fit
    :param model:  dictionary model of fit parameters
    :param baseline_points: the number of points at the start of the FID to ignore
    :return: Dictionary containing ["model": optimized model, "fit": fitting data, "err": dictionary of standard errors]
    """

    # Get list of metabolite names.
    def get_metabolites(model_input):
        metabolites = []
        for name, value in model_input.items():
            if type(value) is dict:
                metabolites.append(name)
        return metabolites

    # Get standard errors from lmfit MinimizerResult object.
    def get_errors(result):
        errors = {}
        for name, param in result.params.items():
            errors[name] = param.stderr
        return errors

    def phase_fid(fid_in, phase0, phase1):
        """
        This function performs a Fourier Transform on the FID to shift it into phase.

        :param fid_in: FID to be fitted.
        :param phase1: phase1 value.
        :param phase0: phase0 value.
        :return: FID that has been shifted into phase by FFT
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
        # metabolite_name_list = []
        # for param in params.keys():
        #     split = param.split('_')
        #     if len(split) == 2:
        #         if split[0] not in metabolite_name_list:
        #             metabolite_name_list.append(split[0])

        basis_matrix = numpy.matrix(numpy.zeros((len(metabolite_name_list), len(time_axis) * 2)))
        for i, metabolite_name in enumerate(metabolite_name_list):
            gaussian = suspect.basis.gaussian(time_axis,
                                              params["{}_frequency".format(metabolite_name)],
                                              params["{}_phase".format(metabolite_name)].value,
                                              params["{}_width".format(metabolite_name)])
            real_gaussian = complex_to_real(gaussian)
            basis_matrix[i, :] = real_gaussian
        return basis_matrix

    def unphase(data, params):

        unphased_data = phase_fid(data, -params['phase0'], -params['phase1'])
        real_unphased_data = complex_to_real(unphased_data)

        return real_unphased_data

    def do_fit(params, time_axis, real_unphased_data):
        """
        This function performs the fitting.

        :param params: lmfit Parameters object containing fitting parameters.
        :param time_axis: the time axis.
        :param real_unphased_data:
        :return: List of fitted data points and amplitudes of each singlet.
        """
        basis = make_basis(params, time_axis)

        weights = scipy.optimize.nnls(basis[:, baseline_points:-baseline_points].T,
                                      real_unphased_data[baseline_points:-baseline_points])[0]

        fitted_data = numpy.array(numpy.dot(weights, basis)).squeeze()
        return fitted_data, weights

    def residual(params, time_axis, data):
        """
        This function calculates the residual to be minimized by the least squares means method.

        :param params: lmfit Parameters object containing fitting parameters.
        :param time_axis: the time axis.
        :param data: FID to be fitted.
        :return: residual values of baseline points.
        """

        real_unphased_data = unphase(data, params)
        fitted_data, weights = do_fit(params, time_axis, real_unphased_data)
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
        fitting_result = lmfit.minimize(residual,
                                        initial_params,
                                        args=(data.time_axis(), data),
                                        xtol=5e-3)

        real_fitted_data, fitting_weights = do_fit(fitting_result.params, data.time_axis(), unphase(data, fitting_result.params))
        fitted_data = real_to_complex(real_fitted_data)

        return fitting_weights, fitted_data, fitting_result

    # Convert lmfit parameters to model format
    def parameters_to_model(parameters_obj, param_weights):
        # metabolite_name_list = []
        # for param in parameters_obj.keys():
        #     split = param.split('_')
        #     if len(split) == 2:
        #         if split[0] not in metabolite_name_list:
        #             metabolite_name_list.append(split[0])
        # Create dictionary for new model.
        new_model = {}
        for param_name, param in parameters_obj.items():
            name = param_name.split("_")
            name1 = name[0]
            if len(name) == 1:  # i.e. phase
                new_model[name1] = param.value
            else:
                name2 = name[1]
                if name1 not in new_model:
                    new_model[name1] = {name2: param.value}
                else:
                    new_model[name1][name2] = param.value

        for i, metabolite_name in enumerate(metabolite_name_list):
            new_model[metabolite_name]["amplitude"] = param_weights[i]

        return new_model

    # Convert initial model to lmfit parameters.
    def model_to_parameters(model_dict):
        lmfit_parameters = lmfit.Parameters()
        params = []
        ordered_params = []
        # Calculate dependencies/references for each parameter.
        depend_dict = calculate_dependencies(model_dict)

        # Construct lmfit Parameter input for each parameter.
        for name1, value1 in model_dict.items():
            if isinstance(value1, numbers.Number):  # (e.g. phase0)
                params.append((name1, value1))
            if type(value1) is dict:
                # Fix phase value to 0 by default.
                if "phase" not in value1:
                    params.append(("{0}_{1}".format(name1, "phase"), None, None, None, None, "0"))
                for name2, value2 in value1.items():
                    # Initialize lmfit parameter arguments.
                    name = "{0}_{1}".format(name1, name2)
                    value = None
                    vary = True
                    lmfit_min = None
                    lmfit_max = None
                    expr = None
                    if isinstance(value2, numbers.Number):
                        value = value2
                    elif isinstance(value2, str):
                        expr = value2
                    elif isinstance(value2, dict):
                        if "value" in value2:
                            value = value2["value"]
                        if "min" in value2:
                            lmfit_min = value2["min"]
                        if "max" in value2:
                            lmfit_max = value2["max"]
                    # Add parameter object with defined parameters.
                    params.append((name, value, vary, lmfit_min, lmfit_max, expr))  # (lmfit Parameter input format)

        # Order parameters based on dependencies.
        in_oparams = []
        while len(params) > 0:
            front = params.pop(0)
            name = front[0]
            # If no dependencies, add parameter to list and mark parameter as added.
            if name not in depend_dict or depend_dict[name] is None:
                ordered_params.append(front)
                in_oparams.append(name)
            else:
                dependencies_present = True
                for dependency in depend_dict[name]:
                    # If dependency not yet added, mark parameter to move to back of queue.
                    if dependency not in in_oparams:
                        dependencies_present = False
                # If all dependencies present, add parameter to list and mark parameter as added.
                if dependencies_present:
                    ordered_params.append(front)
                    in_oparams.append(name)
                # If dependencies missing, move parameter to back of queue.
                else:
                    params.append(front)

        # Convert all parameters to lmfit Parameter objects.
        lmfit_parameters.add_many(*ordered_params)

        return lmfit_parameters

    # Check if all model input types are correct.
    def check_errors(check_model):
        # Allowed keys in the model.
        allowed_keys = ["min", "max", "value", "phase", "amplitude"]

        # Scan model.
        for name1, value1 in check_model.items():
            if not isinstance(value1, (numbers.Number, dict)):
                raise TypeError("Value of {0} must be a number (for phases), or a dictionary.".format(name1))
            elif type(value1) is dict:  # i.e. type(value) is not int
                for name2, value2 in value1.items():
                    if not isinstance(value2,(numbers.Number,dict,str)):
                        raise TypeError("Value of {0}_{1} must be a value, an expression, or a dictionary."
                                        .format(name1, name2))
                    if type(value2) is dict:
                        for key in value2:
                            # Dictionary must have 'value' key.
                            if "value" not in value2:
                                raise KeyError("Dictionary {0}_{1} is missing 'value' key.".format(name1, name2))
                            # Dictionary can only have 'min,' 'max,' and 'value'.
                            if key not in allowed_keys:
                                raise KeyError("In {0}_{1}, '{2}' is not an allowed key.".format(name1, name2, key))

    # Calculate references to determine order for Parameters.
    def calculate_dependencies(unordered_model):
        dependencies = {}  # (name, [dependencies])

        # Compile dictionary of effective names.
        for name1, value1 in unordered_model.items():
            if type(value1) is dict:  # i.e. not phase
                for name2 in value1:
                    dependencies["{0}_{1}".format(name1, name2)] = None

        # Find dependencies for each effective name.
        for name1, value1 in unordered_model.items():
            if type(value1) is dict:  # i.e. not phase
                for name2, value2 in value1.items():
                    if type(value2) is str:
                        lmfit_name = "{0}_{1}".format(name1, name2)
                        dependencies[lmfit_name] = []
                        for depend in dependencies:
                            if depend in value2:
                                dependencies[lmfit_name].append(depend)
        print(dependencies)

        # Check for circular dependencies.
        for name, dependents in dependencies.items():
            if type(dependents) is list:
                for dependent in dependents:
                    #print(dependent)
                    #print(dependencies)
                    if dependencies[dependent] is not None and name in dependencies[dependent]:
                        raise ReferenceError("{0} and {1} reference each other, creating a circular reference."
                                             .format(name, dependent))
        return dependencies

    # Do singlet fitting
    # Minimize and fit 31P data.

    check_errors(model)  # Check for errors in model formatting.

    metabolite_name_list = get_metabolites(model)  # Set list of metabolite names.

    parameters = model_to_parameters(model)  # Convert model to lmfit Parameters object.

    fitted_weights, fitted_data, fitted_results = fit_data(fid, parameters)  # Fit data.

    final_model = parameters_to_model(fitted_results.params, fitted_weights)  # Convert fit parameters to model format.

    stderr = get_errors(fitted_results)  # Get stderr values for each parameter.

    return_dict = {"model": final_model, "fit": fitted_data, "errors": stderr}  # Compile output into a dictionary.
    return return_dict

