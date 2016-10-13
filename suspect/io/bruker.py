from suspect import MRSData
from ._common import complex_array_from_iter

import os
import re
import struct


def load_svs_bruker(fid_filename, acqp_filename=None, method_filename=None):
    """
    Load SVS data in the Bruker format

    Parameters
    ----------
    fid_filename: str
        The location of the file containing the fid data to load
    acqp_filename: str, optional
        The location of the acquisition parameters file. If not provided
        a file named acqp in the same directory as fid_filename will be
        used or an error raised if no such file exists.
    method_filename: str, optional
        The location of the method file. If not provided a file named
        method in the same directory as fid_filename will be used, or an
        error raised if no such file exists.

    Returns
    -------
    data: MRSData
        Data read from the file
    """

    if acqp_filename is None:
        path, fid_file = os.path.split(fid_filename)
        acqp_filename = os.path.join(path, "acqp")
    if not os.path.isfile(acqp_filename):
        raise FileNotFoundError("No acqp file found at {0}".format(acqp_filename))

    if method_filename is None:
        path, fid_file = os.path.split(fid_filename)
        method_filename = os.path.join(path, "method")
    if not os.path.isfile(method_filename):
        raise FileNotFoundError("No method file found")

    # read the data out of the method file
    with open(method_filename) as fin:
        method_string = fin.read()
    dwell_time_string = re.search(r"\$PVM_DigDw=\d+\.?\d*", method_string).group()
    # dwell time is in ms, convert to s
    dt = float(dwell_time_string.split("=")[1]) * 1e-3
    digitiser_delay_string = re.search(r"\$PVM_DigShift=\d+", method_string).group()
    digitiser_delay = int(digitiser_delay_string.split("=")[1])

    # read the data out of the acqp file
    with open(acqp_filename) as fin:
        acqp_string = fin.read()
    f0_string = re.search(r"\$BF1=\d+\.\d*", acqp_string).group()
    f0 = float(f0_string.split("=")[1])

    # read the fid data
    with open(fid_filename, 'rb') as fin:
        fid_bytes = fin.read()
    # bruker data is stored as real/imaginary int 32 pairs
    num_ints = len(fid_bytes) // 4
    fid_ints = struct.unpack("{0}i".format(num_ints), fid_bytes)
    data = complex_array_from_iter(iter(fid_ints), num_ints // 2, chirality=-1)

    return MRSData(data[digitiser_delay:], dt, f0)
