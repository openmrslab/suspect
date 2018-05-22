from suspect import MRSSpectrum

import subprocess
import parse
import re
import numpy as np
import os


def save_dpt(filename, data):
    with open(filename, 'wb') as fout:
        fout.write("Dangerplot_version\t1.0\n".encode())
        fout.write("Number_of_points\t{}\n".format(data.np).encode())
        fout.write("Sampling_frequency\t{0:8.8e}\n".format(1.0 / data.dt).encode())
        fout.write("Transmitter_frequency\t{0:8.8e}\n".format(data.f0 * 1e6).encode())
        fout.write("Phi0\t{0:8.8e}\n".format(0).encode())
        fout.write("Phi1\t{0:8.8e}\n".format(0).encode())
        fout.write("PPM_reference\t{0:8.8e}\n".format(data.ppm0).encode())
        fout.write("Echo_time\t{0:8.8e}\n".format(data.te * 1e-3).encode())
        fout.write("Real_FID\tImag_FID\t\n".encode())
        for x in data:
            fout.write("{0.real:8.8e} {0.imag:8.8e}\n".format(x).encode())


def read_output(filename):
    """Reads in a Tarquin txt results file and returns a dict of the information

    Parameters
    ----------
    filename : txt file
        The filename to read from

    Returns
    -------
    result : dict

    """
    with open(filename) as fin:
        data = fin.read()

        metabolite_fits = {}
        fit_quality = {}

        sections = data.split("\n\n")

        # first section is the metabolite concentrations
        metabolite_lines = sections[0].splitlines()[2:]
        for line in metabolite_lines:
            name, concentration, pc_sd, sd = line.split()
            metabolite_fits[name] = {
                "concentration": concentration,
                "sd": pc_sd,
            }

        # second section is the fit quality
        lines = sections[1].splitlines()[2:]
        for line in lines:
            key, value = line.split(":")
            fit_quality[key.strip()] = float(value.strip())

        return {
            "metabolite_fits": metabolite_fits,
            "quality": fit_quality
        }


def read_fit_file(filename):
    """Reads in a Tarquin txt fit file and returns all the fitted plots
    
    Parameters
    ----------
    filename : txt file
        The filename to read from

    Returns
    -------
    result : dict 
    """
    with open(filename) as fin:
        contents = fin.read()

    # we don't know the exact size and shape of the grid of voxels
    # we are reading in yet, so start by making a dictionary of
    # positions and arrange them later
    voxel_dict = {}
    metabolite_names = None

    # the file is divided into data from each voxel sequentially
    # we want to split the file into separate voxel components
    # each voxel data starts with Row : x, Col : y ...
    # use re.split rather than str.split to avoid getting an
    # empty string before the first Row, by matching the new line
    # followed by Row instead.
    for voxel_string in re.split("\n(?=Row)", contents):

        # first two lines contain metadata, the rest is spectral points
        position_string, metabolite_string, data_string = voxel_string.split("\n", 2)
        position = parse.parse("Row : {:d}, Col : {:d}, Slice : {:d}", position_string)

        # metabolite names are only necessary once
        # also, the first four are fixed (ppm, data, fit, baseline)
        # only collect the rest
        if metabolite_names is None:
            metabolite_names = [name.strip() for name in metabolite_string.split(",")[4:]]

        voxel_dict[position] = np.array([[float(j) for j in i.split(',')] for i in data_string.splitlines()])

    # get the maximum row, col and slice
    max_row = max([pos[0] for pos in voxel_dict.keys()])
    max_col = max([pos[1] for pos in voxel_dict.keys()])
    max_slice = max([pos[2] for pos in voxel_dict.keys()])

    voxel_shape = voxel_dict.values().__iter__().__next__().shape
    combined_data = np.zeros((max_row, max_col, max_slice, *voxel_shape), 'complex')
    for (row, col, slc), voxel_data in voxel_dict.items():
        combined_data[row - 1, col - 1, slc - 1] = voxel_data

    return metabolite_names, combined_data


def _extract_fit_data(ref_data, metabolite_names, combined_data):
    # helper function to save space
    def make_spectrum(spectrum_data):
        return MRSSpectrum(spectrum_data,
                           ref_data.dt,
                           ref_data.f0,
                           ref_data.te,
                           ref_data.ppm0)

    # split out the individual lines (data, fit, baseline etc.)
    data = make_spectrum(combined_data[:, :, :, :, 1].squeeze())
    fit = make_spectrum(combined_data[:, :, :, :, 2].squeeze())
    baseline = make_spectrum(combined_data[:, :, :, :, 3].squeeze())
    metabolite_dict = {
        metabolite_name: make_spectrum(combined_data[:, :, :, :, i + 4].squeeze())
        for i, metabolite_name in enumerate(metabolite_names)
    }
    return {
        "data": data,
        "fit": fit,
        "baseline": baseline,
        "metabolites": metabolite_dict
    }


def process(data, wref=None, options={}):
    save_dpt("/tmp/temp.dpt", data)
    if wref is not None:
        save_dpt("/tmp/wref.dpt", wref)
        options["input_w"] = "/tmp/wref.dpt"
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
        result = read_output("/tmp/output.txt")
    else:
        raise FileNotFoundError("Could not find TARQUIN output file at /tmp/output.txt")
    metabolite_names, fit_data = read_fit_file("/tmp/fit.txt")
    fit_results = _extract_fit_data(data, metabolite_names, fit_data)
    result["plots"] = fit_results
    return result
