import lmfit
import numpy
import scipy.optimize
import numbers
import copy

import suspect.basis


def complex_to_real(complex_fid):
    """This function takes a complex FID and constructs a real version.

    Standard optimization routines as used in lmfit require real data.
    Concatenates the imaginary part to the complex part.
    The imaginary part is also reversed to keep the maxima at each end of the FID
    and avoid discontinuities in the center.

    Parameters
    ----------
    complex_fid : MRSData instance
        The complex FID to be converted to real.

    Returns
    -------
    real_fid : MRSData instance
        The real FID, which has twice as many points as the input.

    """
    np = complex_fid.shape[0]
    real_fid = numpy.zeros(np * 2)
    real_fid[:np] = complex_fid.real
    real_fid[np:] = complex_fid.imag[::-1]
    return real_fid


def real_to_complex(real_fid):
    """This function takes a real FID and converts it back into a true complex form.

    Standard optimization routines as used in lmfit require real data.

    Parameters
    ----------
    real_fid : MRSData instance
        The real FID to be converted to complex, generated from the optimization routine.

    Returns
    -------
    complex_fid : MRSData instance
        The complex version of the FID.

    """
    np = int(real_fid.shape[0] / 2)
    complex_fid = numpy.zeros(np, 'complex')
    complex_fid[:] = real_fid[:np]

    # the imaginary part of the FID has been reversed, have to flip it back
    imag_fid = real_fid[np:]
    complex_fid += 1j * imag_fid[::-1]
    return complex_fid


def fit(fid, model, baseline_points=16):
    """Fit fid with model parameters.

    Parameters
    ----------
    fid : MRSData instance
        FID to be fit
    model : dict
        Dictionary of model fit parameters
    baseline_points : int
        The number of points at the start of the FID to ignore

    Returns
    -------
    dict
        Dictionary containing ["model": optimized model, "fit": fitting data, "err": dictionary of standard errors]

    """
    # Get list of metabolite names.
    def get_metabolites(model_input):
        metabolites = []
        for fid_property_name, fid_property_value in model_input.items():
            if type(fid_property_value) is dict:
                metabolites.append(fid_property_name)
        return metabolites

    # Get standard errors from lmfit MinimizerResult object.
    def get_errors(result):
        errors = {}
        for name, param in result.params.items():
            errors[name] = param.stderr
        return errors

    def make_basis(params, time_axis):
        """This function generates a basis set.

        Parameters
        ----------
        params: lmfit Parameters object
             Contains fitting parameters
        time_axis : arange ndarray
            The time axis in ms

        Returns
        -------
        basis_matrix : matrix
            A 2D array containing the generated basis set.

        """
        basis_matrix = numpy.matrix(numpy.zeros((len(metabolite_name_list), len(time_axis) * 2)))
        for i, metabolite_name in enumerate(metabolite_name_list):
            gaussian = suspect.basis.gaussian(time_axis,
                                              params["{}_frequency".format(metabolite_name)],
                                              params["{}_phase".format(metabolite_name)].value,
                                              params["{}_fwhm".format(metabolite_name)])
            real_gaussian = complex_to_real(gaussian)
            basis_matrix[i, :] = real_gaussian
        return basis_matrix

    # unphase the FID to make it pure absorptive
    def unphase(data, params):

        unphased_data = suspect.adjust_phase(data, -params['phase0'], -params['phase1'])
        real_unphased_data = complex_to_real(unphased_data)

        return real_unphased_data

    def do_fit(params, time_axis, real_unphased_data):
        """This function performs the fitting.

        Parameters
        ----------
        params: lmfit Parameters object
            Contains fitting parameters.
        time_axis : arange ndarray
            The time axis in ms
        real_unphased_data : MRSData instance
            Unphased FID

        Returns
        -------
        list
            Contains fitted data points and amplitudes of each singlet.

        """
        basis = make_basis(params, time_axis)

        weights = scipy.optimize.nnls(basis[:, baseline_points:-baseline_points].T,
                                      real_unphased_data[baseline_points:-baseline_points])[0]

        fitted_data = numpy.array(numpy.dot(weights, basis)).squeeze()
        return fitted_data, weights

    def residual(params, time_axis, data):
        """Calculates the residual to be minimized by the least squares means method.

        Parameters
        ----------
        params : lmfit Parameters object
            Contains fitting parameters
        time_axis : array
            The time axis in ms
        data : MRSData instance
            FID to be fitted

        Returns
        -------
        array
            Residual values

        """
        real_unphased_data = unphase(data, params)
        fitted_data, weights = do_fit(params, time_axis, real_unphased_data)
        res = fitted_data - real_unphased_data

        return res[baseline_points:-baseline_points]

    def fit_data(data, initial_params):
        """
        This function takes an FID and a set of parameters contained in an lmfit Parameters object,
        and fits the data using the least squares means method.

        Parameters
        ----------
        data : MRSData instance
            FID to be fitted.
        initial_params : lmfit Parameters object
            Contains initial fitting parameters.

        Returns
        -------
        fitting_weights : list
            Tuple of weights
        fitted_data : MRSData
            Fitted data
        fitting_result : lmfit MinimizerResult instance

        """
        fitting_result = lmfit.minimize(residual,
                                        initial_params,
                                        args=(data.time_axis(), data),
                                        xtol=5e-3)

        real_fitted_data, fitting_weights = do_fit(fitting_result.params, data.time_axis(), unphase(data, fitting_result.params))
        fitted_data = data.inherit(real_to_complex(real_fitted_data))

        return fitting_weights, fitted_data, fitting_result

    # Convert lmfit parameters to model format
    def parameters_to_model(parameters_obj, param_weights):
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

        model_dict_copy = copy.deepcopy(model_dict)
        params.append(("phase0", model_dict_copy.pop("phase0")))
        params.append(("phase1", model_dict_copy.pop("phase1")))

        # Construct lmfit Parameter input for each parameter.
        for peak_name, peak_properties in model_dict_copy.items():
                # Fix phase value to 0 by default.
                if "phase" not in peak_properties:
                    params.append(("{0}_{1}".format(peak_name, "phase"), None, None, None, None, "0"))
                for property_name, property_value in peak_properties.items():
                    # Initialize lmfit parameter arguments.
                    name = "{0}_{1}".format(peak_name, property_name)
                    value = None
                    vary = True
                    lmfit_min = None
                    lmfit_max = None
                    expr = None
                    if isinstance(property_value, numbers.Number):
                        value = property_value
                    elif isinstance(property_value, str):
                        expr = property_value
                    elif isinstance(property_value, dict):
                        if "value" in property_value:
                            value = property_value["value"]
                        if "min" in property_value:
                            lmfit_min = property_value["min"]
                        if "max" in property_value:
                            lmfit_max = property_value["max"]
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
        for model_property, model_values in check_model.items():
            if not isinstance(model_values, (numbers.Number, dict)):
                raise TypeError("Value of {0} must be a number (for phases), or a dictionary.".format(model_property))
            elif type(model_values) is dict:  # i.e. type(value) is not int
                for peak_property, peak_value in model_values.items():
                    if not isinstance(peak_value, (numbers.Number, dict, str)):
                        raise TypeError("Value of {0}_{1} must be a value, an expression, or a dictionary."
                                        .format(model_property, peak_property))
                    if type(peak_value) is dict:
                        for width_param in peak_value:
                            # Dictionary must have 'value' key.
                            if "value" not in peak_value:
                                raise KeyError("Dictionary {0}_{1} is missing 'value' key."
                                               .format(model_property, peak_property))
                            # Dictionary can only have 'min,' 'max,' and 'value'.
                            if width_param not in allowed_keys:
                                raise KeyError("In {0}_{1}, '{2}' is not an allowed key."
                                               .format(model_property, peak_property, width_param))

    # Calculate references to determine order for Parameters.
    def calculate_dependencies(unordered_model):
        dependencies = {}  # (name, [dependencies])

        # Compile dictionary of effective names.
        for model_property, model_values in unordered_model.items():
            if type(model_values) is dict:  # i.e. pcr, not phase
                for peak_property in model_values:
                    dependencies["{0}_{1}".format(model_property, peak_property)] = None

        # Find dependencies for each effective name.
        for model_property, model_values in unordered_model.items():
            if type(model_values) is dict:  # i.e. not phase
                for peak_property, peak_value in model_values.items():
                    if type(peak_value) is str:
                        lmfit_name = "{0}_{1}".format(model_property, peak_property)
                        dependencies[lmfit_name] = []
                        for depend in dependencies:
                            if depend in peak_value:
                                dependencies[lmfit_name].append(depend)

        # Check for circular dependencies.
        for name, dependents in dependencies.items():
            if type(dependents) is list:
                for dependent in dependents:
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
