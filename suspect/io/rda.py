from suspect import MRSData, transformation_matrix

import numpy
import struct
import re

# The RDA format consists of a large number of key value pairs followed by raw
# data. The values need to be cast into different datatypes depending on the
# key, this dictionary stores a mapping of key to datatype.

rda_types = {
    "floats": ["PatientWeight", "TR", "TE", "TM", "TI", "DwellTime", "NumberOfAverages",
               "MRFrequency", "MagneticFieldStrength", "FlipAngle", "SliceThickness",
               "FoVHeight", "FoVWidth", "PercentOfRectFoV", "PixelSpacingRow",
               "PixelSpacingCol", "VOIPositionSag", "VOIPositionCor",
               "VOIPositionTra", "VOIThickness", "VOIPhaseFOV", "VOIReadoutFOV",
               "VOIReadoutVOV", "VOINormalSag", "VOINormalCor", "VOINormalTra",
               "VOIRotationInPlane", "FoV3D", "PixelSpacing3D"],
    "integers": ["SeriesNumber", "InstanceNumber", "AcquisitionNumber", "NumOfPhaseEncodingSteps",
                 "NumberOfRows", "NumberOfColumns", "VectorSize", "EchoNumber",
                 "NumberOf3DParts", "HammingFilterWidth", "NumberOfEchoes"],
    "strings": ["PatientID", "PatientName", "StudyDescription", "PatientBirthDate",
                "StudyDate", "StudyTime", "PatientAge", "SeriesDate", "SeriesTime",
                "SeriesDescription", "ProtocolName", "PatientPosition", "ModelName",
                "StationName", "InstitutionName", "DeviceSerialNumber", "InstanceDate",
                "InstanceTime", "InstanceComments", "SequenceName", "SequenceDescription",
                "Nucleus", "TransmitCoil", "PatientSex", "HammingFilter", "FrequencyCorrection"],
    "float_arrays": ["PositionVector", "RowVector", "ColumnVector"],
    "integer_arrays": ["CSIMatrixSize", "CSIMatrixSizeOfScan", "CSIGridShift"],
    "string_arrays": ["SoftwareVersion"],
    "dictionaries": ["TransmitRefAmplitude"]
}


def load_rda(filename):
    header_dict = {}
    with open(filename, 'rb') as fin:
        header_line = fin.readline().strip()
        if header_line != b">>> Begin of header <<<":
            raise Exception("Error reading file {} as a .rda".format(filename))
        header_line = fin.readline().strip().decode('windows-1252')
        while header_line != ">>> End of header <<<":
            key, value = map(str.strip, header_line.split(":", 1))
            if key in rda_types["strings"]:
                header_dict[key] = value
            elif key in rda_types["integers"]:
                header_dict[key] = int(value)
            elif key in rda_types["floats"]:
                header_dict[key] = float(value)
            elif "[" in key and "]" in key:
                # could be a dict or a list
                key, index = re.split(r"\]|\[", key)[0:2]
                if key in rda_types["dictionaries"]:
                    if key not in header_dict:
                        header_dict[key] = {}
                    header_dict[key][index] = value
                else:
                    # not a dictionary, must be a list
                    if key in rda_types["float_arrays"]:
                        value = float(value)
                    elif key in rda_types["integer_arrays"]:
                        value = int(value)
                    index = int(index)
                    # make sure there is a list in the header_dict, with enough entries
                    if not key in header_dict:
                        header_dict[key] = []
                    while len(header_dict[key]) <= index:
                        header_dict[key].append(0)
                    header_dict[key][index] = value
            header_line = fin.readline().strip().decode('windows-1252')
        # now we can read the data
        data = fin.read()

    # the shape of the data in slice, column, row, time format
    data_shape = header_dict["CSIMatrixSize"][::-1]
    data_shape.append(header_dict["VectorSize"])
    data_shape = numpy.array(data_shape)
    data_size = numpy.prod(data_shape) * 16  # each data point is a complex double, 16 bytes
    if data_size != len(data):
        raise ValueError("Error reading file {}: expected {} bytes of data, got {}".format(filename, data_size, len(data)))

    # unpack the data into complex numbers
    data_as_floats = struct.unpack("<{}d".format(numpy.prod(data_shape) * 2), data)
    float_iter = iter(data_as_floats)
    complex_iter = (complex(r, i) for r, i in zip(float_iter, float_iter))
    complex_data = numpy.fromiter(complex_iter, "complex64", int(numpy.prod(data_shape)))
    complex_data = numpy.reshape(complex_data, data_shape).squeeze()

    # some .rda files have a misnamed field, correct this here
    if "VOIReadoutFOV" not in header_dict:
        if "VOIReadoutVOV" in header_dict:
            header_dict["VOIReadoutFOV"] = header_dict.pop("VOIReadoutVOV")

    # combine positional elements in the header
    voi_size = (header_dict["VOIReadoutFOV"],
                header_dict["VOIPhaseFOV"],
                header_dict["VOIThickness"])
    voi_center = (header_dict["VOIPositionSag"],
                  header_dict["VOIPositionCor"],
                  header_dict["VOIPositionTra"])
    voxel_size = (header_dict["PixelSpacingCol"],
                  header_dict["PixelSpacingRow"],
                  header_dict["PixelSpacing3D"])

    x_vector = numpy.array(header_dict["RowVector"])
    y_vector = numpy.array(header_dict["ColumnVector"])

    to_scanner = transformation_matrix(x_vector, y_vector, numpy.array(voi_center), voxel_size)

    # put useful components from the header in the metadata
    metadata = {
        "voi_size": voi_size,
        "position": voi_center,
        "voxel_size": voxel_size,
        "protocol": header_dict["ProtocolName"],
        "to_scanner": to_scanner,
        "from_scanner": numpy.linalg.inv(to_scanner)
    }

    return MRSData(complex_data,
                   header_dict["DwellTime"] * 1e-6,
                   header_dict["MRFrequency"],
                   te=header_dict["TE"],
                   tr=header_dict["TR"],
                   transform=to_scanner,
                   metadata=metadata)
