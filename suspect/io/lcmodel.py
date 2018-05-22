import numpy
import os
import itertools
import parsley
import warnings

basis_grammar = r"""
namelist = '$' name:n pairs:p ws -> (n, dict(p))
pairs = (pair:first (pair)*:rest -> [first] + rest) | -> []
pair = ws name:k ws '=' ws valuelist:v ws ','? -> (k.upper(), v)
valuelist = ws (stringlist|numberlist|truth|falsehood):v -> v
truth = 'T' -> True
falsehood = 'F' -> False
stringlist = (string:first (ws string)+:rest -> [first] + rest) | string
string = '\'' (~'\'' anything)*:c '\'' -> ''.join(c).strip()
numberlist = (number:first (ws number)+:rest -> [first] + rest) | number
number = ('-' | -> ''):sign (digits:ds (floatPart(sign ds) | -> int(sign + ds)))
digits = <digit+>
name = <letter (letter | digit | '_')*>
floatPart :sign :ds = <('.' digits? exponent?) | exponent>:tail -> float(sign + ds + tail)
exponent = ('e' | 'E') ('+' | '-')? digits
"""

parser = parsley.makeGrammar(basis_grammar, {})


def save_raw(filename, data):
    with open(filename, 'w') as fout:
        fout.write(" $SEQPAR\n")
        fout.write(" ECHOT = {}\n".format(data.te))
        fout.write(" HZPPPM = {}\n".format(data.f0))
        fout.write(" SEQ = 'PRESS'\n")
        fout.write(" $END\n")
        fout.write(" $NMID\n")
        fout.write(" FMTDAT = '(2E15.6)'\n")
        # if we don't know the volume, just let LCModel use its default
        # convert the volume from mm^3 to cc
        if data.transform is not None:
            fout.write(" VOLUME = {}\n".format(data.voxel_volume() * 1e-3))
        else:
            warnings.warn("Saving LCModel data without a transform, using default voxel volume of 1ml")
        fout.write(" $END\n")
        for point in numpy.nditer(data, order='C'):
            fout.write("  {0: 4.6e}  {1: 4.6e}\n".format(float(point.real), float(point.imag)))


def write_all_files(filename, data, wref_data=None, params=None):
    """
    Creates an LCModel control file for processing the supplied MRSData, and
    optional water reference data, updating the default parameters with any
    values supplied through params.

    Parameters
    ----------
    filename :  directory location
        the location where the control file should be saved.
    data : MRSData instance
        MRSData to be processed.
    wref_data : MRSData instance
        Optional MRSData containing water reference.
    params : dict
        Optional dictionary containing non-default parameter values.

    """

    # we assume that the data has one spectral dimension, any others must be
    # spatial
    if len(data.shape) == 1:
        shape = (1, 1, 1)
    elif len(data.shape) == 2:
        shape = (data.shape[0], 1, 1)
    elif len(data.shape) == 3:
        shape = (data.shape[0], data.shape[1], 1)
    elif len(data.shape) == 4:
        shape = data.shape[0:3]
    elif len(data.shape) > 4:
        raise ValueError("LCModel cannot handle data with more than 4 dimensions")

    # We need to save a bunch of files for LCModel to process: a raw file for
    # the data, possibly a raw file for the wref and a control file for each
    # slice. In addition, in the absence of information in the params file
    # about where to save the output (.ps, .csv, .table etc.) that should also
    # be saved in the same folder as the input data for LCModel.
    folder, file_root = os.path.split(filename)

    # make sure that the folder exists before trying to save things to it
    if not os.path.isdir(folder):
        os.makedirs(folder)

    file_root, ext = os.path.splitext(file_root)

    base_params = {
        "FILBAS": "/home/spectre/.lcmodel/basis-sets/provencher/press_te30_3t_gsh_v3.basis",
        "ICOLST": 1,
        "ICOLEN": shape[0],
        "NDCOLS": shape[0],
        "IROWST": 1,
        "IROWEN": shape[1],
        "NDROWS": shape[1],
        "NDSLIC": shape[2],
        "DOWS": True if wref_data is not None else False,
        "DOECC": True if wref_data is not None else False,
        "FILRAW": os.path.join(folder, file_root + ".RAW"),
        "FILPS": os.path.join(folder, file_root + ".PS")
    }
    if wref_data is not None:
        base_params["FILH2O"] = os.path.join(folder, file_root + ".H2O")

    # add the user supplied parameters to the list
    if params is not None:
        base_params.update(params)

    # make a few modifications based on user edits
    if "FILTAB" in base_params:
        base_params["LTABLE"] = 7
        base_params["FILTAB"] = "{}".format(base_params["FILTAB"])
    elif "LTABLE" in base_params:
        base_params["LTABLE"] = 7
        base_params["FILTAB"] = "{}".format(os.path.join(folder, file_root + ".TABLE"))
    if "FILCSV" in base_params:
        base_params["LCSV"] = 11
        base_params["FILCSV"] = "{}".format(base_params["FILCSV"])
    elif "LCSV" in base_params:
        base_params["LCSV"] = 11
        base_params["FILCSV"] = "{}".format(os.path.join(folder, file_root + ".CSV"))
    if "FILCOO" in base_params:
        base_params["LCOORD"] = 9
        base_params["FILCOO"] = "{}".format(base_params["FILCOO"])
    elif "LCOORD" in base_params:
        base_params["LCOORD"] = 9
        base_params["FILCOO"] = "{}".format(os.path.join(folder, file_root + ".COORD"))
    if "FILCOR" in base_params:
        base_params["LCORAW"] = 10
        base_params["FILCOR"] = "{}".format(base_params["FILCOR"])
    elif "LCORAW" in base_params:
        base_params["LCORAW"] = 10
        base_params["FILCOR"] = "{}".format(os.path.join(folder, file_root + ".CORAW"))

    save_raw(base_params["FILRAW"], data)
    if wref_data is not None:
        save_raw(base_params["FILH2O"], wref_data)
    # have to add single quotes to the various paths
    base_params["FILRAW"] = "{}".format(base_params["FILRAW"])
    base_params["FILBAS"] = "{}".format(base_params["FILBAS"])
    base_params["FILPS"] = "{}".format(base_params["FILPS"])
    if wref_data is not None:
        base_params["FILH2O"] = "{}".format(base_params["FILH2O"])

    for slice_index in range(shape[2]):
        control_filename = "{0}_sl{1}.CONTROL".format(file_root, slice_index)
        control_filepath = os.path.join(folder, control_filename)
        with open(control_filepath, 'wt') as fout:
            fout.write(" $LCMODL\n")
            fout.write(" OWNER = ''\n")
            fout.write(" KEY = 123456789\n")
            fout.write(" DELTAT = {}\n".format(data.dt))
            fout.write(" HZPPPM = {}\n".format(data.f0))
            fout.write(" NUNFIL = {}\n".format(data.np))
            for key, value in base_params.items():
                if isinstance(value, str):
                    value = "'{0}'".format(value)
                elif isinstance(value, bool):
                    value = 'T' if value else 'F'
                fout.write(" {0} = {1}\n".format(key, value))
            fout.write(" $END\n")


def read_coord(filename):
    with open(filename, 'rt') as fin:
        coord_lines = fin.readlines()

    # find start of metabolite table
    for index, line in enumerate(coord_lines):
        if "lines in following concentration table" in line:
            break
    metabolite_table_info_line = index
    # unfortunately LCModel sometimes lies to us about how many metabolites are
    # in the table, so the line below does not work
    # metabolite_table_line_count = int(coord_lines[metabolite_table_info_line].split()[0])
    # instead we have to look for the start of the next section and use that to
    # work out how many lines are left

    for index, line in enumerate(coord_lines):
        if "following misc. output table" in line:
            break
    misc_output_info_line = index
    metabolite_table_line_count = misc_output_info_line - 1 - metabolite_table_info_line

    metabolite_table_lines = coord_lines[(metabolite_table_info_line + 2):(metabolite_table_info_line + 1 + metabolite_table_line_count)]
    metabolite_fits = {}
    for metabolite_line in metabolite_table_lines:
        conc = float(metabolite_line[:9])
        sd = int(metabolite_line[9:13])
        cr_ratio = float(metabolite_line[14:22])
        name = metabolite_line[22:].strip()
        metabolite_fits[name] = {
            "concentration": conc,
            "sd": sd,
        }
    # add some aliases for some of the composite metabolites
    if "NAA+NAAG" in metabolite_fits:
        metabolite_fits["TNAA"] = metabolite_fits["NAA+NAAG"]
    if "GPC+PCh" in metabolite_fits:
        metabolite_fits["TCho"] = metabolite_fits["GPC+PCh"]
    if "Cr+PCr" in metabolite_fits:
        metabolite_fits["TCr"] = metabolite_fits["Cr+PCr"]
    if "Glu+Gln" in metabolite_fits:
        metabolite_fits["Glx"] = metabolite_fits["Glu+Gln"]

    # because LCModel lies to us about the number of metabolites we have to get
    # the position of the misc output a different way higher up
    # misc_output_info_line = metabolite_table_info_line + metabolite_table_line_count + 1
    misc_output_line_count = int(coord_lines[misc_output_info_line].split()[0])
    misc_output_lines = coord_lines[(misc_output_info_line + 1):(misc_output_info_line + misc_output_line_count + 1)]
    misc_output = {
        "fwhm": float(misc_output_lines[0].split()[2]),
        "snr": float(misc_output_lines[0].split()[6]),
        "frequency_shift": float(misc_output_lines[1].split("=")[1].split()[0]),
        "phase0": float(misc_output_lines[2].split()[1]),
        "phase1": float(misc_output_lines[2].split()[3])
    }

    # get the ppm axis
    ppm_axis_info_line = misc_output_info_line + misc_output_line_count + 1
    num_ppm_points = int(coord_lines[ppm_axis_info_line].split()[0])
    # there are always 10 points of data per line
    number_of_point_lines = int((num_ppm_points + 9) / 10)

    def read_points(starting_line):
        points = []
        # we can use map to convert the 10 string numbers on each line to a
        # list of floats, then extend the current list with the new one
        for line_number in range(starting_line, starting_line + number_of_point_lines):
            points.extend(map(float, coord_lines[line_number].split()))
        return points

    ppm_points = read_points(ppm_axis_info_line + 1)

    # after ppm axis is the phased data
    data_info_line = ppm_axis_info_line + number_of_point_lines + 1
    data_points = read_points(data_info_line + 1)

    # after data is the fit
    fit_info_line = data_info_line + number_of_point_lines + 1
    fit_points = read_points(fit_info_line + 1)

    # after the fit is the baseline
    baseline_info_line = fit_info_line + number_of_point_lines + 1
    baseline_points = read_points(baseline_info_line + 1)

    # now come 0 or more metabolite spectra, followed by the diagnostic table
    metabolite_spectra = {}
    metabolite_info_line = baseline_info_line + number_of_point_lines + 1
    while "diagnostic table" not in coord_lines[metabolite_info_line]:
        metabolite_name = coord_lines[metabolite_info_line].split()[0]
        metabolite_points = read_points(metabolite_info_line + 1)
        metabolite_spectra[metabolite_name] = metabolite_points

        # move to the next block of data
        metabolite_info_line = metabolite_info_line + number_of_point_lines + 1

    return {
        "metabolite_fits": metabolite_fits,
        "misc_output": misc_output,
        "ppm": ppm_points,
        "fit": fit_points,
        "data": data_points,
        "baseline": baseline_points,
        "metabolite_spectra": metabolite_spectra
    }


def read_basis(filename):
    with open(filename) as fin:
        basis_set = {"SPECTRA": {}}
        data = fin.read().lstrip()
        # does the data start with a namelist
        while data.startswith("$"):
            # where does the namelist end
            namelist_data, _, data = data.partition("$END")
            data = data.lstrip()
            namelist = parser(namelist_data).namelist()

            if namelist[0].upper() == "SEQPAR":
                basis_set["SEQPAR"] = namelist[1]

            elif namelist[0].upper() == "BASIS1":
                basis_set["BASIS1"] = namelist[1]
                # find out the number of points in each spectrum
                np = namelist[1]["NDATAB"]
                # 3 complex points per line, how many lines
                num_lines = ((np - 1) // 3) + 1

            elif namelist[0].upper() == "BASIS":
                metabolite_name = namelist[1]["METABO"]
                basis_set["SPECTRA"][metabolite_name] = namelist[1]
                # after each BASIS namelist come the actual data points
                points = []
                for line in itertools.islice(data.splitlines(), num_lines):
                    points.extend(map(float, line.split()))

                points = iter(points)

                complex_points = (complex(r, i) for r, i in zip(points, points))
                complex_data = numpy.fromiter(complex_points, "complex64", np)
                basis_set["SPECTRA"][metabolite_name]["data"] = complex_data
                # find start of next namelist
                start = data.find("$")
                data = data[start:]

        return basis_set


def save_basis(filename, basis):
    # make sure that the basis object has the necessary components
    if "BASIS1" not in basis:
        raise ValueError("Basis object {} is missing required component BASIS1".format(basis))
    if len(basis["SPECTRA"]) == 0:
        raise ValueError("Basis object {} contains zero spectra".format(basis))

    with open(filename, 'wt') as fout:

        # write SEQPAR if it is in the basis
        if "SEQPAR" in basis:
            write_namelist(fout, "SEQPAR", basis["SEQPAR"])

        # write BASIS1
        write_namelist(fout, "BASIS1", basis["BASIS1"])

        # write the metabolite namelists and data
        for metabolite, properties in basis["SPECTRA"].items():

            # remove the data object (the spectrum itself)
            data = properties.pop("data")
            # write the remainder of the basis namelist to the file
            write_namelist(fout, "BASIS", properties)
            for i, point in enumerate(data):
                fout.write(" {0: 4.5e} {1: 4.5e}".format(point.real, point.imag))
                if (i + 1) % 3 == 0:
                    fout.write("\n")
            if (i+1) % 3 != 0:
                fout.write("\n")


def write_namelist(fout, name, components):
    fout.write(" ${}\n".format(name))
    for key, value in components.items():
        if isinstance(value, str):
            value = "'{0}'".format(value)
        fout.write(" {0} = {1},\n".format(key, value))
    fout.write(" $END\n")
