from suspect import MRSData

import os
import numpy

spar_types = {
    "floats": ["ap_size", "lr_size", "cc_size", "ap_off_center", "lr_off_center",
               "cc_off_center", "ap_angulation", "lr_angulation", "cc_angulation",
               "image_plane_slice_thickness", "slice_distance", "spec_col_lower_val",
               "spec_col_upper_val", "spec_row_lower_val", "spec_row_upper_val",
               "spectrum_echo_time", "echo_time"],
    "integers": ["samples", "rows", "synthesizer_frequency", "offset_frequency",
                 "sample_frequency", "echo_nr", "mix_number", "t0_mul_direction",
                 "repetition_time", "averages", "volumes",
                 "volume_selection_method", "nr_of_slices_for_multislice",
                 "spec_num_col", "spec_num_row", "num_dimensions", "TSI_factor",
                 "spectrum_inversion_time", "image_chemical_shift",
                 "t0_mu1_direction"],
    "strings": ["scan_id", "scan_date", "patient_name", "patient_birth_date",
                "patient_position", "patient_orientation", "nucleus",
                "volume_selection_enable", "phase_encoding_enable", "t1_measurement_enable",
                "t2_measurement_enable", "time_series_enable", "Spec.image in plane transf",
                "spec_data_type", "spec_sample_extension", "spec_col_extension",
                "spec_row_extension", "echo_acquisition", "resp_motion_comp_technique",
                "de_coupling", "equipment_sw_verions", "examination_name"],
}


def load_sdat(sdat_filename, spar_filename=None, spar_encoding=None):
    # if the spar filename is not supplied, assume it is in the same folder as
    # the sdat and only differs in the extension
    if spar_filename is None:
        path, ext = os.path.splitext(sdat_filename)
        # match the capitalisation of the sdat extension
        if ext == ".SDAT":
            spar_filename = path + ".SPAR"
        elif ext == ".sdat":
            spar_filename = path + ".spar"

    with open(spar_filename, 'r', encoding=spar_encoding) as fin:
        parameter_dict = {}
        for line in fin:
            # ignore empty lines and comments starting with !
            if line != "\n" and not line.startswith("!"):
                key, value = map(str.strip, line.split(":", 1))
                if key in spar_types["floats"]:
                    parameter_dict[key] = float(value)
                elif key in spar_types["integers"]:
                    parameter_dict[key] = int(value)
                elif key in spar_types["strings"]:
                    parameter_dict[key] = value
                else:
                    pass
                    #print("{} : {}".format(key, value))

    dt = 1 / parameter_dict["sample_frequency"]

    with open(sdat_filename, 'rb') as fin:
        raw_bytes = fin.read()

    floats = _vax_to_ieee_single_float(raw_bytes)
    data_iter = iter(floats)
    complex_iter = (complex(r, -i) for r, i in zip(data_iter, data_iter))
    raw_data = numpy.fromiter(complex_iter, "complex64")
    raw_data = numpy.reshape(raw_data, (parameter_dict["rows"], parameter_dict["samples"])).squeeze()
    return MRSData(raw_data,
                   dt,
                   parameter_dict["synthesizer_frequency"] * 1e-6,
                   te=parameter_dict["echo_time"],
                   tr=parameter_dict["repetition_time"])


def _vax_to_ieee_single_float(data):
    """Converts a float in Vax format to IEEE format.

    Data should be a single string of chars that have been read in from
    a binary file. These will be processed 4 at a time into float values.
    Thus the total number of byte/chars in the string should be divisible
    by 4.

    Notes
    -----
    Based on VAX data organization in a byte file, we need to do a bunch of
    bitwise operations to separate out the numbers that correspond to the
    sign, the exponent and the fraction portions of this floating point
    number

    role :      S        EEEEEEEE      FFFFFFF      FFFFFFFF      FFFFFFFF
    bits :      1        2      9      10                               32
    bytes :     byte2           byte1               byte4         byte3

    Returns
    -------
    f : array
        Contains floats in IEEE format

    """
    f = []
    nfloat = int(len(data) / 4)
    for i in range(nfloat):

        byte2 = data[0 + i*4]
        byte1 = data[1 + i*4]
        byte4 = data[2 + i*4]
        byte3 = data[3 + i*4]

        # hex 0x80 = binary mask 10000000
        # hex 0x7f = binary mask 01111111

        sign = (byte1 & 0x80) >> 7
        expon = ((byte1 & 0x7f) << 1) + ((byte2 & 0x80) >> 7)
        fract = ((byte2 & 0x7f) << 16) + (byte3 << 8) + byte4

        if sign == 0:
            sign_mult = 1.0
        else:
            sign_mult = -1.0

        if 0 < expon:
            # note 16777216.0 == 2^24
            val = sign_mult * (0.5 + (fract/16777216.0)) * pow(2.0, expon - 128.0)
            f.append(val)
        elif expon == 0 and sign == 0:
            f.append(0)
        else:
            f.append(0)
            # may want to raise an exception here ...

    return f
