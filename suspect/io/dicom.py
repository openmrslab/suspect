from suspect import MRSData
import numpy as np

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
    dataset = pydicom.dcmread(filename)

    # format for metadata dictionary elements:
    #   {'key': [[dicom_tag], required], ...}
    metadata = {'sw': [[0x0018, 0x9052], 'True'],
                'f0': [[0x0018, 0x9098], 'True'],
                'te': [[0x0018, 0x0081], 'False'],
                'tr': [[0x0018, 0x0080], 'False'],
                'ppm0': [[0x0018, 0x9053], 'True'],
                'rows': [[0x0028, 0x0010], 'True'],
                'cols': [[0x0028, 0x0011], 'True'],
                'frames': [[0x0028, 0x0008], 'True'],
                'num_second_spectral': [[0x0028, 0x9001], 'True'],
                'num_points': [[0x0028, 0x9002], 'True']
                }

    parameters = {}

    for key in metadata:
        try:
            parameters[key] = dataset[metadata[key][0]].value
        except KeyError:
            if metadata[key][1] == 'True':
                raise KeyError("Missing required DICOM tag - {0}: {1}".format(key, metadata[key][0]))
            else:
                parameters[key] = None

    parameters['dt'] = 1.0 / parameters['sw']

    data_shape = [parameters['frames'], parameters['rows'], parameters['cols'], parameters['num_second_spectral'],
                  parameters['num_points']]

    # versions of pydicom >2.0.0 require explicit conversion from bytestring to list
    if type(dataset[0x5600, 0x0020].value) == bytes:
        data_iter = iter(np.fromstring(dataset[0x5600, 0x0020].value, dtype=np.float32))

    elif type(dataset[0x5600, 0x0020].value) == list:
        data_iter = iter(dataset[0x5600, 0x0020].value)

    else:
        raise TypeError("Unknown data type for dataset[0x5600, 0x0020].value")

    data = complex_array_from_iter(data_iter, shape=data_shape, chirality=-1)

    return MRSData(data,
                   parameters['dt'],
                   parameters['f0'],
                   te=parameters['te'],
                   tr=parameters['tr'],
                   ppm0=parameters['ppm0'])
