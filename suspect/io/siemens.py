import pydicom.tag
import pydicom.dicomio
import numpy
import struct
import warnings

from suspect import MRSData, transformation_matrix, rotation_matrix
from ._common import complex_array_from_iter

CSA1 = 0
CSA2 = 1

ima_types = {
    "floats": ["NumberOfAverages", "RSatPositionSag", "PercentPhaseFieldOfView", "RSatOrientationSag", "MixingTime",
               "PercentPhaseFieldOfView", "RSatPositionCor", "InversionTime", "RepetitionTime", "VoiThickness",
               "TransmitterReferenceAmplitude", "ImageOrientationPatient", "SliceThickness", "RSatOrientationTra",
               "PixelBandwidth", "SAR", "PixelSpacing", "ImagePositionPatient", "VoiPosition", "SliceLocation",
               "FlipAngle", "VoiInPlaneRotation", "VoiPhaseFoV", "SliceMeasurementDuration", "HammingFilterWidth",
               "RSatPositionTra", "MagneticFieldStrength", "VoiOrientation", "PercentSampling", "EchoTime",
               "VoiReadoutFoV", "RSatThickness", "RSatOrientationCor", "ImagingFrequency", "TriggerTime", "dBdt",
               "TransmitterCalibration", "PhaseGradientAmplitude", "ReadoutGradientAmplitude",
               "SelectionGradientAmplitude", "GradientDelayTime", "dBdt_max", "t_puls_max", "dBdt_thresh",
               "dBdt_limit", "SW_korr_faktor", "Stim_lim", "Stim_faktor"],
    "integers": ["Rows", "Columns", "DataPointColumns", "SpectroscopyAcquisitionOut-of-planePhaseSteps",
                 "EchoPartitionPosition", "AcquisitionMatrix", "NumberOfFrames", "EchoNumbers", "RealDwellTime",
                 "EchoTrainLength", "EchoLinePosition", "EchoColumnPosition", "SpectroscopyAcquisitionDataColumns",
                 "SpectroscopyAcquisitionPhaseColumns", "SpectroscopyAcquisitionPhaseRows", "RfWatchdogMask",
                 "NumberOfPhaseEncodingSteps", "DataPointRows", "UsedPatientWeight", "NumberOfPrescans",
                 "Stim_mon_mode", "Operation_mode_flag", "CoilId", "MiscSequenceParam", "MrProtocolVersion",
                 "ProtocolSliceNumber"],
    "strings": ["ReferencedImageSequence", "ScanningSequence", "SequenceName", "ImagedNucleus", "TransmittingCoil",
                "PhaseEncodingDirection", "VariableFlipAngleFlag", "SequenceMask", "AcquisitionMatrixText",
                "MultistepIndex", "DataRepresentation", "SignalDomainColumns", "k-spaceFiltering", "ResonantNucleus",
                "ImaCoilString", "FrequencyCorrection", "WaterReferencedPhaseCorrection", "SequenceFileOwner",
                "CoilForGradient", "CoilForGradient2", "PositivePCSDirections", ],
}


def read_csa_header(csa_header_bytes):
    # two possibilities exist here, either this is a CSA2 format beginning with an SV10 string, or a CSA1 format which
    # doesn't. in CSA2 after the "SV10" are four junk bytes, then the number of tags in a uint32 and a delimiter uint32
    # containing the value 77. in CSA1 there is just the number of tags and the delimiter. after that the two formats
    # contain the same structure for each tag, but the definition of the size of the items in each tag is different
    # between the two versions
    if csa_header_bytes[:4] == "SV10".encode('latin-1'):
        num_tags, delimiter = struct.unpack("<II", csa_header_bytes[8:16])
        header_offset = 16
        header_format = CSA2
    else:
        num_tags, delimiter = struct.unpack("<II", csa_header_bytes[:8])
        header_offset = 8
        header_format = CSA1
    # now we can iteratively read the tags and the items inside them
    csa_header = {}
    for i in range(num_tags):
        name, vm, vr, syngo_dt, nitems, delimiter = struct.unpack("<64si4siii",
                                                                  csa_header_bytes[header_offset:(header_offset + 84)])
        header_offset += 84
        # the name of the tag is 64 bytes long, but the string we want is null-terminated inside, so extract the
        # real name by taking only bytes up until the first 0x00
        name = name.decode('latin-1')
        name = name.split("\x00", 1)[0]
        # read all the items inside this tag
        item_list = []
        for j in range(nitems):
            sizes = struct.unpack("<4L", csa_header_bytes[header_offset:(header_offset + 16)])
            header_offset += 16
            if header_format == CSA2:
                item_length = sizes[1]
                if (header_offset + item_length) > len(csa_header_bytes):
                    item_length = len(csa_header_bytes) - header_offset
            elif header_format == CSA1:
                item_length = sizes[0]
            item, = struct.unpack("<%ds" % item_length,
                                  csa_header_bytes[header_offset:(header_offset + item_length)])
            item = item.decode('latin-1')
            item = item.split("\x00", 1)[0]
            if item_length > 0:
                if name in ima_types["floats"]:
                    item = float(item)
                elif name in ima_types["integers"]:
                    item = int(item)
                elif name in ima_types["strings"]:
                    pass
                else:
                    warnings.warn("Unhandled name {0} with vr {1} and value {2}".format(name, vr, item))
                item_list.append(item)
            header_offset += item_length
            header_offset += (4 - (item_length % 4)) % 4  # move the offset to the next 4 byte boundary
        if len(item_list) == 1:
            item_list = item_list[0]
        csa_header[name] = item_list
    return csa_header


def load_siemens_dicom(filename):
    """Imports a file in the Siemens .IMA format.

    Parameters
    ----------
    filename : str
        The name of the file to import

    """
    # the .IMA format is a DICOM standard, unfortunately most of the information is contained inside a private and very
    # complicated header with its own data storage format, we have to get that information out along with the data
    # start by reading in the DICOM file completely
    dataset = pydicom.dicomio.read_file(filename)
    # now look through the tags (0029, 00xx) to work out which xx refers to the csa header
    # xx seems to start at 10 for Siemens
    xx = 0x0010
    header_index = 0
    while (0x0029, xx) in dataset:
        if dataset[0x0029, xx].value == "SIEMENS CSA HEADER":
            header_index = xx
        xx += 1
    # check that we have found the header
    if header_index == 0:
        raise KeyError("Could not find header index")
    # now we know which tag contains the CSA image header info: (0029, xx10)
    csa_header_bytes = dataset[0x0029, 0x0100 * header_index + 0x0010].value
    csa_header = read_csa_header(csa_header_bytes)
    # for key, value in csa_header.items():
    #    print("%s : %s" % (str(key), str(value)))
    # we can also get the series header info: (0029, xx20), but this seems to be mostly pretty boring

    # now we can work out the shape of the data (slices, rows, columns, fid_points)
    data_shape = (csa_header["SpectroscopyAcquisitionOut-of-planePhaseSteps"],
                  csa_header["Rows"],
                  csa_header["Columns"],
                  csa_header["DataPointColumns"],
                  )

    # now look through the tags (0029, 00xx) to work out which xx refers to the csa header
    # xx seems to start at 10 for Siemens
    xx = 0x0010
    data_index = 0
    while (0x7fe1, xx) in dataset:
        if dataset[0x7fe1, xx].value == "SIEMENS CSA NON-IMAGE":
            data_index = xx
        xx += 1
    # check that we have found the data
    if data_index == 0:
        raise KeyError("Could not find data index")
    # extract the actual data bytes
    csa_data_bytes = dataset[0x7fe1, 0x0100 * data_index + 0x0010].value
    # the data is stored as a list of 4 byte floats in (real, imaginary) pairs
    data_floats = struct.unpack("<%df" % (len(csa_data_bytes) / 4), csa_data_bytes)

    # a bug report (#143) has been submitted that for at least one .IMA dataset
    # created with an old Siemens VB17 WIP, the data_shape worked out above
    # does not match the actual size of the data because the
    # Out-of-planePhaseSteps value is not the number of slices. Assuming this
    # is a rare situation that is unlikely to happen often, the simple solution
    # is simply to check the size matches here, and if not then use the size
    # of data available as the shape
    available_points = len(data_floats) // 2
    if numpy.prod(data_shape) != available_points:
        data_shape = (available_points,)
        warnings.warn("The calculated data shape for this file {} does not "
                      "match the size of data contained in the file {}. "
                      "Therefore the returned data shape from this function "
                      "will simply be ({},), any reshaping must be done by "
                      "the user. If you need help with this or believe this "
                      "has occured in error, please raise an issue at"
                      "https://github.com/openmrslab/suspect/issues.")

    complex_data = complex_array_from_iter(iter(data_floats),
                                           length=len(data_floats) // 2,
                                           shape=data_shape)

    in_plane_rot = csa_header["VoiInPlaneRotation"]
    x_vector = numpy.array([-1, 0, 0])
    normal_vector = numpy.array(csa_header["VoiOrientation"])
    orthogonal_x = x_vector - numpy.dot(x_vector, normal_vector) * normal_vector
    orthonormal_x = orthogonal_x / numpy.linalg.norm(orthogonal_x)
    rot_matrix = rotation_matrix(in_plane_rot, normal_vector)
    row_vector = numpy.dot(rot_matrix, orthonormal_x)
    column_vector = numpy.cross(row_vector, normal_vector)
    voxel_size = (*csa_header["PixelSpacing"],
                  csa_header["SliceThickness"])
    transform = transformation_matrix(row_vector,
                                      column_vector,
                                      csa_header["VoiPosition"],
                                      voxel_size)

    voi_size = [csa_header["VoiReadoutFoV"],
                csa_header["VoiPhaseFoV"],
                csa_header["VoiThickness"]]

    metadata = {
        "voi_size": voi_size
    }

    return MRSData(complex_data,
                   csa_header["RealDwellTime"] * 1e-9,
                   csa_header["ImagingFrequency"],
                   te=csa_header["EchoTime"],
                   tr=csa_header["RepetitionTime"],
                   transform=transform,
                   metadata=metadata)


# def anonymize_siemens_dicom(filename, anonymized_filename):
# TODO: anonymize dicom
#     """
#     Anonymizes an MRS data file in Siemens IMA DICOM format.
#
#     :param filename:
#     :param anonymized_filename:
#     :return:
#     """
#     dataset = pydicom.dicomio.read_file(filename)
#     print(dataset.PatientName)
#     xx = 0x0010
#     header_index = 0
#     for i in range(4624):
#         if pydicom.tag.Tag(0x0029, i) in dataset:
#             print(pydicom.tag.Tag(0x0029, i))
#             print(dataset[(0x0029, i)].value)
#     xx = 0x0010
#     header_index = 0
#     while (0x0029, xx) in dataset:
#         if dataset[0x0029, xx].value == "SIEMENS CSA HEADER":
#             header_index = xx
#         xx += 1
#     # check that we have found the header
#     if header_index == 0:
#         raise KeyError("Could not find header index")
#     # now we know which tag contains the CSA image header info: (0029, xx10)
#     csa_header_bytes = dataset[0x0029, 0x0100 * header_index + 0x0010].value
#     csa_header = read_csa_header(csa_header_bytes)
#     csa_series_header = dataset[0x0029, 0x0100 * header_index + 0x0020].value
#     csa_series = read_csa_header(csa_series_header)
