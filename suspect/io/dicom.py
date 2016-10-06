from suspect import MRSData

from ._common import complex_array_from_iter

import pydicom.dicomio
import pydicom.tag


def load_dicom(filename):
    """
    Load a file in the DICOM Magnetic Resonance Spectroscopy format
    (SOP 1.2.840.10008.5.1.4.1.1.4.2)

    Parameters
    ----------
    filename : str
        The name of the file to load

    Returns
    -------
    MRSData
        The loaded data from the file
    """
    dataset = pydicom.dicomio.read_file(filename)

    sw = dataset[0x0018, 0x9052].value
    dt = 1.0 / sw

    f0 = dataset[0x0018, 0x9098].value

    ppm0 = dataset[0x0018, 0x9053].value

    rows = dataset[0x0028, 0x0010].value
    cols = dataset[0x0028, 0x0011].value
    frames = dataset[0x0028, 0x0008].value
    num_second_spectral = dataset[0x0028, 0x9001].value
    num_points = dataset[0x0028, 0x9002].value

    data_shape = [frames, rows, cols, num_second_spectral, num_points]

    # turn the data into a numpy array
    data_iter = iter(dataset[0x5600, 0x0020])
    data = complex_array_from_iter(data_iter, shape=data_shape, chirality=-1)

    return MRSData(data,
                   dt,
                   f0,
                   ppm0=ppm0)
