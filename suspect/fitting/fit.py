"""
File Name: fit.py
Author: Sam Jiang
Purpose: fit_31p() function fits single 31P fid.
Interpreter: Anaconda 3.5.1

fit(fid, model)
Input: fid and model in hierarchical dictionary format
Output: dictionary with 'model', 'fit', and 'error' keys and corresponding values
"""

import os
import numpy
import json
import lmfit
import array
import suspect


# Load model in.
def load(fin):
    if not os.path.isfile(fin):
        raise FileNotFoundError("{} not found.".format(fin))
    with open(fin, 'r+') as f:
        model = json.load(f)
    return model


# Save model or fit out.
def save(output, fout):
    # Check if file and directory exists.
    directory, name = os.path.split(fout)
    if directory == '':
        directory = os.getcwd()
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if name == '':
        raise IsADirectoryError("{} is not a file name.".format(name))
    ftype = os.path.splitext(name)[1]

    # If model, save as json.
    if type(output) is dict:
        if ftype != ".json":
            raise TypeError("Output file must be a JSON (.json) file.")

        with open(fout, 'w+') as f:
            json.dump(output, f)

    # If fit, save as csv or txt.
    elif type(output) is array.ArrayType:
        if ftype != ".csv" or ftype != ".txt":
            raise TypeError("Output file must be a CSV (.csv) or Text (.txt) file.")

        with open(fout, 'w+') as f:
            first = True
            for x in output.tolist():
                if first:
                    f.write("{}".format(x))
                    first = False
                else:
                    f.write(", {}".format(x))


# Fit fid
# Input: fid, model
# Output: dictionary containing optimized model, fit data, standard errors
def fit(fid, model):
    # MODEL
    # Get list of metabolite names
    # def get_metabolites(model_input):
    #     metabolites = []
    #     for name, value in model_input.items():
    #         if type(value) is dict:
    #             metabolites.append(name)
    #     return metabolites

    # Get standard errors from lmfit MinimizerResult object.
    def get_errors(result):
        errors = {}
        for name, param in result.params.items():
            errors[name] = param.stderr
        return errors

    # Convert lmfit parameters to model format
    def parameters_to_model(parameters_obj, param_weights):
        metabolite_name_list = []
        for param in parameters_obj.keys():
            split = param.split('_')
            if len(split) == 2:
                if split[0] not in metabolite_name_list:
                    metabolite_name_list.append(split[0])
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
            if type(value1) is int:  # (e.g. phase0)
                params.append((name1, value1))
            if type(value1) is dict:
                # Fix phase value to 0 by default.
                if "phase" not in value1:
                    params.append(("{}_{}".format(name1, "phase"), None, None, None, None, "0"))
                for name2, value2 in value1.items():
                    # Initialize lmfit parameter arguments.
                    name = "{}_{}".format(name1, name2)
                    value = None
                    vary = None
                    lmfit_min = None
                    lmfit_max = None
                    expr = None
                    if type(value2) is int:
                        value = value2
                    elif type(value2) is str:
                        expr = value2
                    if type(value2) is dict:
                        if "value" in value2:
                            value = value2["value"]
                        # if "vary" in value2:
                        #     vary = value2["vary"]
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
        # Allowed names and keys in the model.
        allowed_names = ["pcr", "atpc", "atpb", "atpa", "pi", "pme", "pde", "phase0", "phase1"]
        allowed_keys = ["min", "max", "value", "phase", "amplitude"]

        # Scan model.
        for name1, value1 in check_model.items():
            if type(value1) is not int and type(value1) is not float and type(value1) is not dict:
                raise TypeError("Value of {} must be a number (for phases), or a dictionary.".format(name1))
            elif name1 not in allowed_names:
                raise NameError("{} is not an allowed name.".format(name1))
            elif type(value1) is dict:  # i.e. type(value) is not int
                for name2, value2 in value1.items():
                    if type(value2) is not int and type(value2) is not float and type(value2) is not dict and \
                                    type(value2) is not str:
                        raise TypeError("Value of {}_{} must be a value, an expression, or a dictionary."
                                        .format(name1, name2))
                    if type(value2) is dict:
                        for key in value2:
                            # Dictionary must have 'value' key.
                            if "value" not in value2:
                                raise KeyError("Dictionary {}_{} is missing 'value' key.".format(name1, name2))
                            # Dictionary can only have 'min,' 'max,' and 'value'.
                            if key not in allowed_keys:
                                raise KeyError("In {}_{}, '{}' is not an allowed key.".format(name1, name2, key))

        return

    # Calculate references to determine order for Parameters.
    def calculate_dependencies(unordered_model):
        dependencies = {}  # (name, [dependencies])

        # Compile dictionary of effective names.
        for name1, value1 in unordered_model.items():
            if type(value1) is dict:  # i.e. not phase
                for name2 in value1:
                    dependencies["{}_{}".format(name1, name2)] = None

        # Find dependencies for each effective name.
        for name1, value1 in unordered_model.items():
            if type(value1) is dict:  # i.e. not phase
                for name2, value2 in value1.items():
                    if type(value2) is str:
                        lmfit_name = "{}_{}".format(name1, name2)
                        dependencies[lmfit_name] = []
                        for depend in dependencies:
                            if depend in value2:
                                dependencies[lmfit_name].append(depend)

        # Check for circular dependencies.
        for name, dependents in dependencies.items():
            if type(dependents) is list:
                for dependent in dependents:
                    if name in dependencies[dependent]:
                        raise ReferenceError("{} and {} reference each other, creating a circular reference."
                                             .format(name, dependent))

        return dependencies

    # MAIN
    def main():
        # Minimize and fit 31P data.
        # Check for errors in model formatting.
        check_errors(model)

        # Set list of metabolite names.
        # nonlocal metabolite_name_list
        # metabolite_name_list = get_metabolites(model)

        # Convert model to lmfit Parameters object.
        parameters = model_to_parameters(model)

        # Fit data.
        fitted_weights, fitted_data, fitted_results = suspect.fitting.singlet.fit_data(fid, parameters)

        # Convert fit parameters to model format.
        final_model = parameters_to_model(fitted_results.params, fitted_weights)
        # Get stderr values for each parameter.
        stderr = get_errors(fitted_results)

        # Compile output into a dictionary.
        return_dict = {"model": final_model, "fit": fitted_data, "errors": stderr}

        return return_dict

    if __name__ == "__main__":
        return main()


# Test fit function
def test_fit(_plot):
    # Test file and model
    testfile = "test_timecourse.csv"
    modelfile = "model.json"
    model = load(modelfile)

    # Calculate references to determine order for Parameters.
    def calculate_dependencies(unordered_model):
        dependencies = {}  # (name, [dependencies])

        # Compile dictionary of effective names.
        for name1, value1 in unordered_model.items():
            if type(value1) is dict:  # i.e. not phase
                for name2 in value1:
                    dependencies["{}_{}".format(name1, name2)] = None

        # Find dependencies for each effective name.
        for name1, value1 in unordered_model.items():
            if type(value1) is dict:  # i.e. not phase
                for name2, value2 in value1.items():
                    if type(value2) is str:
                        lmfit_name = "{}_{}".format(name1, name2)
                        dependencies[lmfit_name] = []
                        for depend in dependencies:
                            if depend in value2:
                                dependencies[lmfit_name].append(depend)

        # Check for circular dependencies.
        for name, dependents in dependencies.items():
            if type(dependents) is list:
                for dependent in dependents:
                    if name in dependencies[dependent]:
                        raise ReferenceError("{} and {} reference each other, creating a circular reference."
                                             .format(name, dependent))

        return dependencies

    # Convert initial model to lmfit parameters.
    def model_to_parameters(model_dict):
        lmfit_parameters = lmfit.Parameters()
        params = []
        ordered_params = []
        # Calculate dependencies/references for each parameter.
        depend_dict = calculate_dependencies(model_dict)

        # Construct lmfit Parameter input for each parameter.
        for name1, value1 in model_dict.items():
            if type(value1) is int:  # (e.g. phase0)
                params.append((name1, value1))
            if type(value1) is dict:
                # Fix phase value to 0 by default.
                if "phase" not in value1:
                    params.append(("{}_{}".format(name1, "phase"), None, None, None, None, "0"))
                for name2, value2 in value1.items():
                    # Initialize lmfit parameter arguments.
                    name = "{}_{}".format(name1, name2)
                    value = None
                    vary = None
                    lmfit_min = None
                    lmfit_max = None
                    expr = None
                    if type(value2) is int:
                        value = value2
                    elif type(value2) is str:
                        expr = value2
                    if type(value2) is dict:
                        if "value" in value2:
                            value = value2["value"]
                        # if "vary" in value2:
                        #     vary = value2["vary"]
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

    def fake_spectrum(params, time_axis, weights):
        basis = suspect.fitting.singlet.make_basis(params, time_axis)
        real_data = numpy.array(numpy.dot(weights, basis)).squeeze()
        complex_data = suspect.fitting.singlet.real_to_complex(real_data)
        return suspect.MRSData(complex_data, dt, f0, te=0, ppm0=0)

    # Generate fake noise (Testing only)
    def generate_noise(n, snr):
        # Added (n * 0) to fix unused variable warning.
        noise = (n * 0) + (numpy.random.randn(4096) + 1j * numpy.random.randn(4096)) * 6.4e-6 / snr
        return noise

    # Generate fake fid (Testing only)
    def generate_fid(ground_truth, snr, initial_params):
        metabolite_name_list = []
        for param in initial_params.keys():
            split = param.split('_')
            if len(split) == 2:
                if split[0] not in metabolite_name_list:
                    metabolite_name_list.append(split[0])

        peaks = {"pcr": 1,
                 "atpc": 0.3,
                 "atpb": 0.3,
                 "atpa": 0.3,
                 "pi": 1,
                 "pme": 0.05,
                 "pde": 0.05}
        amps = []
        for metabolite in metabolite_name_list:
            amps.append(peaks[metabolite])

        noisy_sequence = numpy.zeros((len(ground_truth), np), 'complex')
        noisy_sequence = suspect.MRSData(noisy_sequence, dt, f0, te=0, ppm0=0)

        # Populate the noisy_sequence with data.
        generated_fid = fake_spectrum(initial_params, noisy_sequence.time_axis(), amps)
        # [1, 0.3, 0.3, 0.3, 1, 0.05, 0.05]
        noisy_fid = generated_fid + generate_noise(np, snr)

        return generated_fid, noisy_fid

    # Generate fake FID.
    parameters = model_to_parameters(model)
    sw = 6e3
    dt = 1.0 / sw
    np = 4096
    f0 = 49.885802
    fake_fid, noisy_fid = generate_fid(numpy.loadtxt(testfile), 4, parameters)  # using filename, not actual fid

    # Test fit().
    fit_results = fit(fake_fid, model)

    # Plot fit, noisy_fid, and fake_fid.
    if _plot:
        from matplotlib import pyplot
        pyplot.plot(numpy.fft.fftshift(numpy.fft.fft(fit_results["fit"])))
        pyplot.plot(numpy.fft.fftshift(numpy.fft.fft(noisy_fid)))
        pyplot.plot(numpy.fft.fftshift(numpy.fft.fft(fake_fid)))
        pyplot.show()


test = False
plot = True
if test:
    print("Running test.")
    test_fit(plot)
    print("Test complete.")

