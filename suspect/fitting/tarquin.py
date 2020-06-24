from ..io import tarquin

import os
import subprocess


def process(data, wref=None, aq_factor=None, options={}):
    """
    Runs the Tarquin basis set fitting program to determine metabolite
    concentrations.

    Parameters
    ----------
    data : MRSData
        The water suppressed FID data to be fitted.
    wref : MRSData
        Optional water reference file for concentration scaling.
    aq_factor : float
        Absolute quantification factor.
    options : dict
        Set of Tarquin parameters to override.

    Returns
    -------
    dict
        Output from running Tarquin on the data
    """
    tarquin.save_dpt("/tmp/temp.dpt", data)
    if wref is not None:
        tarquin.save_dpt("/tmp/wref.dpt", wref)
        options["input_w"] = "/tmp/wref.dpt"
    if aq_factor is not None:
        options["w_conc"] = 1
        options["w_att"] = aq_factor
    option_string = ""
    for key, value in options.items():
        option_string += " --{} {}".format(key, value)
    result = subprocess.run("tarquin --input {} --format dpt --output_txt {} --output_fit {}{}".format(
        "/tmp/temp.dpt", "/tmp/output.txt", "/tmp/fit.txt", option_string
    ), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="UTF-8")
    if result.returncode != 0:
        raise Exception("Error doing quantification with TARQUIN: {}".format(result.stderr))
    # with open("/tmp/output.txt") as fin:
    #    result = fin.read()
    if os.path.isfile("/tmp/output.txt"):
        result = tarquin.read_output("/tmp/output.txt")
    else:
        raise FileNotFoundError("Could not find TARQUIN output file at /tmp/output.txt")
    metabolite_names, fit_data = tarquin.read_fit_file("/tmp/fit.txt")
    fit_results = tarquin._extract_fit_data(data, metabolite_names, fit_data)
    result["plots"] = fit_results
    return result