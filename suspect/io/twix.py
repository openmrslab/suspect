from contextlib import contextmanager
from suspect import MRSData, transformation_matrix, rotation_matrix

import struct
import numpy
#import quaternion
import re
import io
import os

# This file largely relies on information from Siemens regarding the structure
# of the TWIX file formats. Most of the parameters that are read use the same
# names as those apparently used internally at Siemens. The purpose of some of
# these parameters is not currently clear but we break them out anyway.


class TwixBuilder(object):
    def __init__(self):
        self.header_params = None
        self.dt = None
        self.np = None
        self.num_channels = None
        self.data = []
        self.loop_counters = []

    def set_header_string(self, header_string):
        self.header_params = parse_twix_header(header_string)

    def set_np(self, np):
        if self.np is None:
            self.np = np
        else:
            if self.np != np:
                raise ValueError("TwixBuilder already has an np of {}, now trying to set a different value of {}".format(self.np, np))

    def set_num_channels(self, num_channels):
        if self.num_channels is None:
            self.num_channels = num_channels
        else:
            if self.num_channels != num_channels:
                raise ValueError("TwixBuilder num_channels already set to {}, now being changed to {}".format(self.num_channels, num_channels))

    def add_scan(self, loop_counters, scan_data):
        self.loop_counters.append(loop_counters)
        self.data.append(scan_data)

    def build_mrsdata(self):
        loop_counter_array = numpy.array(self.loop_counters)
        data_shape = 1 + numpy.max(loop_counter_array, axis=0)
        data_shape = numpy.append(data_shape, (self.num_channels, self.np))
        data = numpy.zeros(data_shape, dtype='complex')
        for i, loop_counter in enumerate(loop_counter_array):
            # have to break out the loop_counter parameters individually, the second line should work but doesn't
            data[loop_counter[0], loop_counter[1], loop_counter[2], loop_counter[3], loop_counter[4], loop_counter[5], loop_counter[6], loop_counter[7], loop_counter[8], loop_counter[9], loop_counter[10], loop_counter[11], loop_counter[12], loop_counter[13]] = self.data[i]
            # data[loop_counter] = self.data[i]

        # get rid of all the size 1 dimensions
        data = data.squeeze()

        metadata = {
            "patient_name": self.header_params["patient_name"],
            "patient_id": self.header_params["patient_id"],
            "patient_birthdate": self.header_params["patient_birthdate"],
            "exam_date": self.header_params["exam_date"],
            "exam_time": self.header_params["exam_time"],
        }
        mrs_data = MRSData(data,
                           self.header_params["dt"],
                           self.header_params["f0"],
                           te=self.header_params["te"],
                           tr=self.header_params["tr"],
                           metadata=metadata,
                           transform=self.header_params["transform"])

        return mrs_data


def calculate_orientation(normal):
    """

    Parameters
    ----------
    normal : the normal vector to the image plane

    Returns
    -------

    """
    if normal[2] >= normal[1] - 1e-6 and normal[2] >= normal[0] - 1e-6:
        return "TRA"
    elif normal[1] >= normal[0] - 1e-6:
        return "COR"
    return "SAG"


def get_meta_regex(regex_list, header_string, convert=1, default=None):
    """
    Extract metadata using list of regex and apply unit convertion if value is number.
    If regex found, but value is empty (usually empty string), will return default.

    Parameters
    ----------
    regex_list : list
        List of regex string
    header_string : string
        TWIX header string
    convert : int, optional
        Unit convertion, if value is number. Defaults to 1 (means no convertion)
    default
        Default value if match found, but value is empty. Defaults to None.

    Returns
    -------
    If number, converted to float and converted to unit (by multiplying by convert).
    If non-number, return as is. If empty, return default.

    Raises
    ------
    KeyError: if no matches found.
    """
    for rgx in regex_list:
        match = re.findall(rgx, header_string)
        if len(match) > 0:
            match_string = match[-1]
            try:
                result = float(match_string) * convert
            except (ValueError, TypeError):
                result = match_string if match_string else default
            return result
    raise KeyError("Regex list not found in header string. {}".format(regex_list))


def parse_twix_header(header_string):
    # get the name of the protocol being acquired
    protocol_name_matches = [
        r"tProtocolName\s*=\s*\"(.*)\"\s*"
    ]
    protocol_name = get_meta_regex(protocol_name_matches, header_string)
    # get information about the subject being scanned
    patient_id_string = re.search(r"<ParamString.\"PatientID\">\s*{\s*\".+\"\s*}\n", header_string).group()
    patient_id = patient_id_string.split("\"")[3]
    patient_name = re.escape(re.search(r"(<ParamString.\"PatientName\">\s*{\s*\")(.+)(\"\s*}\n)", header_string).group(2))
    patient_birthday = re.search(r"(<ParamString.\"PatientBirthDay\">\s*{\s*\")(.+)(\"\s*}\n)", header_string).group(2)
    # get the FrameOfReference to get the date and time of the scan
    frame_of_reference = re.search(r"(<ParamString.\"FrameOfReference\">\s*{\s*)(\".+\")(\s*}\n)", header_string).group(2)
    if re.match("x*", frame_of_reference):
        exam_date = "x" * 6
        exam_time = "x" * 6
    else:
        exam_date_time = frame_of_reference.split(".")[10]
        exam_date = exam_date_time[2:8]
        exam_time = exam_date_time[8:14]

    software_version_matches = [
        r"<ParamString.\"tMeasuredBaselineString\">\s*{\s*(\".+\")\s*}\n",
        r"<ParamString.\"tBaselineString\">\s*{\s*(\".+\")\s*}\n",
        r"<ParamString.\"SoftwareVersions\">\s*{\s*(\".+\")\s*}\n"
    ]
    software_version = get_meta_regex(software_version_matches, header_string)

    manufacturer = re.search(r"<ParamString.\"Manufacturer\">\s*{\s*(\".+\")\s*}\n", header_string).group(1)
    manufacturers_model_name = re.search(r"<ParamString.\"ManufacturersModelName\">\s*{\s*(\".+\")\s*}\n", header_string).group(1)
    sequence_name = re.search(r"<ParamString.\"tSequenceFileName\">\s*{\s*(\".+\")\s*}\n", header_string).group(1)

    # get the scan parameters
    frequency_matches = [
        r"sTXSPEC\.asNucleusInfo\[0\]\.lFrequency\s*=\s*([[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamLong.\"Frequency\">  { (\d*)  }",
        r"<ParamDouble.\"MainFrequency\">  { (.+)}\n"
    ]
    frequency = get_meta_regex(frequency_matches, header_string, convert=1e-6)

    dwell_time_matches = [
        r"sRXSPEC\.alDwellTime\[0\]\s*=\s*([[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamLong.\"DwellTimeSig\">  { (\d*)  }",
        r"<ParamDouble.\"DwellTime\">  { (.+)}",
    ]
    dwell_time = get_meta_regex(dwell_time_matches, header_string, convert=1e-9)

    # get TE
    # TE is stored in us, we would prefer to use ms
    te = float(re.search(r"(alTE\[0\]\s*=\s*)(\d+)", header_string).group(2)) / 1000
    # get TR
    tr = float(re.search(r"(alTR\[0\]\s*=\s*)(\d+)", header_string).group(2)) / 1000

    # get voxel size
    ro_fov_matches = [
        r"sSpecPara\.sVoI\.dReadoutFOV\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamDouble.\"VoI_RoFOV\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoI_RoFOV\">\s*{\s*(-?[0-9\.]+)?\s*}"
    ]
    pe_fov_matches = [
        r"sSpecPara\.sVoI\.dPhaseFOV\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamDouble.\"VoI_PeFOV\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoI_PeFOV\">\s*{\s*(-?[0-9\.]+)?\s*}"
    ]
    slice_thickness_matches = [
        r"sSpecPara\.sVoI\.dThickness\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamDouble.\"VoI_SliceThickness\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoI_SliceThickness\">\s*{\s*(-?[0-9\.]+)?\s*}"
    ]
    ro_fov = get_meta_regex(ro_fov_matches, header_string, default=0)
    pe_fov = get_meta_regex(pe_fov_matches, header_string, default=0)
    slice_thickness = get_meta_regex(slice_thickness_matches, header_string, default=0)

    # get position information
    pos_sag_matches = [
        r"sSpecPara\.sVoI\.sPosition\.dSag\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamDouble\.\"VoI_Position_Sag\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiPositionSag\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_Position_Sag\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoiPositionSag\">  { <Precision> \d+ (-?[0-9\.]+)? }",
        r"<ParamMap\.\"sVoI\">[\s\S]*?<ParamMap\.\"sPosition\">[\s\S]*?<ParamDouble\.\"dSag\">{ (-?[0-9\.]+)?\s?}"
    ]
    pos_cor_matches = [
        r"sSpecPara\.sVoI\.sPosition\.dCor\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamDouble\.\"VoI_Position_Cor\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiPositionCor\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_Position_Cor\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoiPositionCor\">  { <Precision> \d+ (-?[0-9\.]+)? }",
        r"<ParamMap\.\"sVoI\">[\s\S]*?<ParamMap\.\"sPosition\">[\s\S]*?<ParamDouble\.\"dCor\">{ (-?[0-9\.]+)?\s?}"
    ]
    pos_tra_matches = [
        r"sSpecPara\.sVoI\.sPosition\.dTra\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*",
        r"<ParamDouble\.\"VoI_Position_Tra\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiPositionTra\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_Position_Tra\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoiPositionTra\">  { <Precision> \d+ (-?[0-9\.]+)? }",
        r"<ParamMap\.\"sVoI\">[\s\S]*?<ParamMap\.\"sPosition\">[\s\S]*?<ParamDouble\.\"dTra\">{ (-?[0-9\.]+)?\s?}"
    ]
    pos_sag = get_meta_regex(pos_sag_matches, header_string, default=0)
    pos_cor = get_meta_regex(pos_cor_matches, header_string, default=0)
    pos_tra = get_meta_regex(pos_tra_matches, header_string, default=0)

    # get orientation information
    in_plane_rot_matches = [
        r"<ParamDouble\.\"VoI_InPlaneRotAngle\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiInPlaneRot\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiInPlaneRot\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_InPlaneRotAngle\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"dInPlaneRot\">\s*{\s*(-?[0-9\.]+)?\s*}"
    ]
    normal_sag_matches = [
        r"sSpecPara\.sVoI\.sNormal\.dSag\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*$",
        r"<ParamDouble\.\"VoI_Normal_Sag\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiNormalSag\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiNormalSag\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_Normal_Sag\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamMap\.\"sVoI\">[\s\S]*?<ParamMap\.\"sNormal\">[\s\S]*?<ParamDouble\.\"dSag\">{ (-?[0-9\.]+)?\s?}"
    ]
    normal_cor_matches = [
        r"sSpecPara\.sVoI\.sNormal\.dCor\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*$",
        r"<ParamDouble\.\"VoI_Normal_Cor\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiNormalCor\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiNormalCor\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_Normal_Cor\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamMap\.\"sVoI\">[\s\S]*?<ParamMap\.\"sNormal\">[\s\S]*?<ParamDouble\.\"dCor\">{ (-?[0-9\.]+)?\s?}"
    ]
    normal_tra_matches = [
        r"sSpecPara\.sVoI\.sNormal\.dTra\s*=\s*(-?[[0-9]*[.]?[0-9]*]{0,})\s*$",
        r"<ParamDouble\.\"VoI_Normal_Tra\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiNormalTra\">  { <Precision> \d+(  -?[0-9\.]+)?  }",
        r"<ParamDouble\.\"VoiNormalTra\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamDouble\.\"VoI_Normal_Tra\">\s*{\s*(-?[0-9\.]+)?\s*}",
        r"<ParamMap\.\"sVoI\">[\s\S]*?<ParamMap\.\"sNormal\">[\s\S]*?<ParamDouble\.\"dTra\">{ (-?[0-9\.]+)?\s?}"
    ]
    in_plane_rot = get_meta_regex(in_plane_rot_matches, header_string, default=0)
    normal_sag = get_meta_regex(normal_sag_matches, header_string, default=0)
    normal_cor = get_meta_regex(normal_cor_matches, header_string, default=0)
    normal_tra = get_meta_regex(normal_tra_matches, header_string, default=0)

    # the orientation is stored in a somewhat strange way - a normal vector and
    # a rotation angle. to get the row vector, we first use Gram-Schmidt to
    # make [-1, 0, 0] (the default row vector) orthogonal to the normal, and
    # then rotate that vector by the rotation angle (which we do here with a
    # quaternion (not any more, quaternion library has issues with Travis)
    normal_vector = numpy.array([normal_sag, normal_cor, normal_tra])
    if calculate_orientation(normal_vector) == "SAG":
        x_vector = numpy.array([0, 0, 1])
    else:
        x_vector = numpy.array([-1, 0, 0])
    orthogonal_x = x_vector - numpy.dot(x_vector, normal_vector) * normal_vector
    orthonormal_x = orthogonal_x / numpy.linalg.norm(orthogonal_x)
    #rotation_quaternion = quaternion.from_rotation_vector(in_plane_rot * normal_vector)
    #row_vector2 = quaternion.rotate_vectors(rotation_quaternion, orthonormal_x)
    rot_matrix = rotation_matrix(in_plane_rot, normal_vector)
    row_vector = numpy.dot(rot_matrix, orthonormal_x)
    column_vector = numpy.cross(row_vector, normal_vector)
    transform = transformation_matrix(row_vector,
                                      column_vector,
                                      [pos_sag, pos_cor, pos_tra],
                                      [ro_fov, pe_fov, slice_thickness])

    return {"protocol_name": protocol_name,
            "patient_name": patient_name,
            "patient_id": patient_id,
            "patient_birthdate": patient_birthday,
            "dt": dwell_time,
            "f0": frequency,
            "transform": transform,
            "te": te,
            "tr": tr,
            "exam_date": exam_date,
            "exam_time": exam_time,
            "sequence_name": sequence_name,
            "software_version": software_version,
            "manufacturer": manufacturer,
            "manufacturers_model_name": manufacturers_model_name
            }


def load_twix_vb(fin, builder):

    # first four bytes are the size of the header
    header_size = struct.unpack("I", fin.read(4))[0]

    # read the rest of the header minus the four bytes we already read
    header = fin.read(header_size - 4)
    # for some reason the last 24 bytes of the header contain some junk that is not a string
    header = header[:-24].decode('latin-1')
    builder.set_header_string(header)

    # the way that vb files are set up we just keep reading scans until the acq_end flag is set

    while True:
        # start by keeping track of where in the file this scan started
        # this will be used to jump to the start of the next scan
        start_position = fin.tell()

        # the first four bytes contain composite information
        temp = struct.unpack("I", fin.read(4))[0]

        # 25 LSBs contain DMA length (size of this scan)
        DMA_length = temp & (2 ** 26 - 1)
        # next we have the "pack" flag bit and the rest is PCI_rx
        # not sure what either of these are for but break them out in case
        pack_flag = (temp >> 25) & 1
        PCI_rx = temp >> 26

        meas_uid, scan_counter, time_stamp, pmu_time_stamp = struct.unpack("IIII", fin.read(16))

        # next long int is actually a lot of bit flags
        # a lot of them don't seem to be relevant for spectroscopy
        eval_info_mask = struct.unpack("Q", fin.read(8))[0]
        acq_end = eval_info_mask & 1
        rt_feedback = eval_info_mask >> 1 & 1
        hp_feedback = eval_info_mask >> 2 & 1
        sync_data = eval_info_mask >> 5 & 1
        raw_data_correction = eval_info_mask >> 10 & 1
        ref_phase_stab_scan = eval_info_mask >> 14 & 1
        phase_stab_scan = eval_info_mask >> 15 & 1
        sign_rev = eval_info_mask >> 17 & 1
        phase_correction = eval_info_mask >> 21 & 1
        pat_ref_scan = eval_info_mask >> 22 & 1
        pat_ref_ima_scan = eval_info_mask >> 23 & 1
        reflect = eval_info_mask >> 24 & 1
        noise_adj_scan = eval_info_mask >> 25 & 1

        if acq_end:
            break

        # if any of these flags are set then we should ignore the scan data
        if rt_feedback or hp_feedback or phase_correction or noise_adj_scan or sync_data:
            fin.seek(start_position + DMA_length)
            continue

        # now come the actual parameters of the scan
        num_samples, num_channels = struct.unpack("HH", fin.read(4))
        builder.set_num_channels(num_channels)

        # the loop counters are a set of 14 shorts which are used as indices
        # for the parameters an acquisition might loop over, including
        # averaging repetitions, COSY echo time increments and CSI phase
        # encoding steps
        # we have no prior knowledge about which counters might loop in a given
        # scan so we have to read in all scans and then sort out the data shape
        loop_counters = struct.unpack("14H", fin.read(28))

        cut_off_data, kspace_centre_column, coil_select, readout_offcentre = struct.unpack("IHHI", fin.read(12))
        time_since_rf, kspace_centre_line_num, kspace_centre_partition_num = struct.unpack("IHH", fin.read(8))

        ice_program_params = struct.unpack("4H", fin.read(8))
        free_params = struct.unpack("4H", fin.read(8))

        # there are some dummy points before the data starts
        num_dummy_points = free_params[0]

        # we want our np to be the largest power of two within the num_samples - num_dummy_points
        np = int(2 ** numpy.floor(numpy.log2(num_samples - num_dummy_points)))
        builder.set_np(np)

        slice_position = struct.unpack("7f", fin.read(28))

        # construct a numpy ndarray to hold the data from all the channels in this scan
        scan_data = numpy.zeros((num_channels, np), dtype='complex')

        # loop over all the channels and extract data
        for channel_index in range(num_channels):
            channel_id, ptab_pos_neg = struct.unpack("Hh", fin.read(4))
            raw_data = struct.unpack("<{}f".format(num_samples * 2), fin.read(num_samples * 4 * 2))
            # turn the raw data into complex pairs
            data_iter = iter(raw_data)
            complex_iter = (complex(r, -i) for r, i in zip(data_iter, data_iter))
            scan_data[channel_index, :] = numpy.fromiter(complex_iter, "complex64", num_samples)[num_dummy_points:(num_dummy_points + np)]

            # the vb format repeats all the header data for each channel in
            # turn, obviously this is redundant so we read all but the channel
            # index from the next header here
            fin.read(124)

        # pass the data from this scan to the builder
        builder.add_scan(loop_counters, scan_data)

        # go to the next scan and the top of the loop
        fin.seek(start_position + DMA_length)


def load_twix_vd(fin, builder):
    twix_id, num_measurements = struct.unpack("II", fin.read(8))
    # vd file can contain multiple measurements, but we only want the MRS
    # assume that the MRS is the last measurement
    measurement_index = num_measurements - 1

    # measurement headers are each 152 bytes at start of file
    fin.seek(8 + 152 * measurement_index)
    meas_id, file_id, offset, length, patient_name, protocol_name = struct.unpack("IIQQ64s64s", fin.read(152))
    # offset points to where the actual data is in the file
    fin.seek(offset)

    # start with the header
    header_size = struct.unpack("I", fin.read(4))[0]
    header = fin.read(header_size - 4)
    header = header.decode('latin-1')
    builder.set_header_string(header)

    # read each scan until we hit the acq_end flag
    while True:

        # read the initial position, combined with DMA_length below that
        # tells us how to get to the start of the next scan
        initial_position = fin.tell()

        # the first four bytes contain some composite information,
        # read in an int and do bit shift magic to get the values
        temp = struct.unpack("I", fin.read(4))[0]
        DMA_length = temp & (2 ** 26 - 1)
        pack_flag = (temp >> 25) & 1
        PCI_rx = temp >> 26
        meas_uid, scan_counter, time_stamp, pmu_time_stamp = struct.unpack("IIII", fin.read(16))
        system_type, ptab_pos_delay, ptab_pos_x, ptab_pos_y, ptab_pos_z, reserved = struct.unpack("HHIIII", fin.read(20))

        # more composite information
        eval_info_mask = struct.unpack("Q", fin.read(8))[0]
        acq_end = eval_info_mask & 1
        rt_feedback = eval_info_mask >> 1 & 1
        hp_feedback = eval_info_mask >> 2 & 1
        sync_data = eval_info_mask >> 5 & 1
        raw_data_correction = eval_info_mask >> 10 & 1
        ref_phase_stab_scan = eval_info_mask >> 14 & 1
        phase_stab_scan = eval_info_mask >> 15 & 1
        sign_rev = eval_info_mask >> 17 & 1
        phase_correction = eval_info_mask >> 21 & 1
        pat_ref_scan = eval_info_mask >> 22 & 1
        pat_ref_ima_scan = eval_info_mask >> 23 & 1
        reflect = eval_info_mask >> 24 & 1
        noise_adj_scan = eval_info_mask >> 25 & 1

        # if acq_end is set, there is no more data
        if acq_end:
            break

        # there are some data frames that contain auxilliary data, we ignore those for now
        if rt_feedback or hp_feedback or phase_correction or noise_adj_scan or sync_data:
            fin.seek(initial_position + DMA_length)
            continue

        num_samples, num_channels = struct.unpack("HH", fin.read(4))
        builder.set_num_channels(num_channels)
        loop_counters = struct.unpack("14H", fin.read(28))
        cut_off_data, kspace_centre_column, coil_select, readout_offcentre = struct.unpack("IHHI", fin.read(12))
        time_since_rf, kspace_centre_line_num, kspace_centre_partition_num = struct.unpack("IHH", fin.read(8))
        slice_position = struct.unpack("7f", fin.read(28))
        ice_program_params = struct.unpack("24H", fin.read(48))
        reserved_params = struct.unpack("4H", fin.read(8))
        fid_start_offset = ice_program_params[4]
        num_dummy_points = reserved_params[0]
        fid_start = fid_start_offset + num_dummy_points
        np = int(2 ** numpy.floor(numpy.log2(num_samples - fid_start)))
        builder.set_np(np)
        application_counter, application_mask, crc = struct.unpack("HHI", fin.read(8))

        # read the data for each channel in turn
        scan_data = numpy.zeros((num_channels, np), dtype='complex')
        for channel_index in range(num_channels):

            # start with the header
            dma_length, meas_uid, scan_counter, sequence_time, channel_id = struct.unpack("III4xI4xH6x", fin.read(32))

            # now the data itself, which consists of num_samples * 4 (bytes per float) * 2 (two floats per complex)
            raw_data = struct.unpack("<{}f".format(num_samples * 2), fin.read(num_samples * 4 * 2))

            # we need to massage the list of floats into a numpy array of complex numbers
            data_iter = iter(raw_data)
            complex_iter = (complex(r, -i) for r, i in zip(data_iter, data_iter))
            scan_data[channel_index, :] = numpy.fromiter(complex_iter, "complex64", num_samples)[fid_start:(fid_start + np)]

        builder.add_scan(loop_counters, scan_data)

        # move the file pointer to the start of the next scan
        fin.seek(initial_position + DMA_length)

def load_twix(source_file, buffering=io.DEFAULT_BUFFER_SIZE):
    """
    Load TWIX data. 

    Parameters
    ----------
    source_file : str or file-like
        File path of TWIX file

    Returns
    -------
    suspect.MRSData

    """
    @contextmanager
    def open_if_filepath(file_or_path, mode='r', **kwargs):
        """Context manager to open a file if argument is string"""
        if isinstance(file_or_path, (str, os.PathLike)):
            f = open(file_or_path, mode, **kwargs)
            try:
                yield f
            finally:
                f.close()
        else:
            yield file_or_path

    with open_if_filepath(source_file, 'rb') as fin:
        # we can tell the type of file from the first two uints in the header
        first_uint, second_uint = struct.unpack("II", fin.read(8))

        # reset the file pointer before giving to specific function
        fin.seek(0)

        # create a TwixBuilder object for the actual loader function to use
        builder = TwixBuilder()

        if first_uint == 0 and second_uint <= 64:
            load_twix_vd(fin, builder)
        else:
            load_twix_vb(fin, builder)

    return builder.build_mrsdata()


def anonymize_twix_header(header_string):
    """Removes the PHI from the supplied twix header and returns the sanitized version.
    This consists of:
    1) Replacing the patient id and name with strings of lower case x
    characters.
    2) Replacing the patient birthday with 19700101
    3) Replacing the patient gender with the number 0
    4) Replacing all digits in the patient age, weight and height with 0s
    5) All references to the date and time of the exam have all alphanumeric
    characters replaced by lower case x. Other characters (notably the period)
    are left unchanged.

    Parameters
    ----------
    header_string : str
        The header string to be anonymized

    Returns
    -------
    str
        The anonymized version of the header.
    """

    patient_name = r"(<ParamString.\"t?Patients?Name\">\s*\{\s*\")(.+)(\"\s*\}\n)"
    patient_id = r"(<ParamString.\"PatientID\">\s*\{\s*\")(.+)(\"\s*\}\n)"
    patient_birthday = r"(<ParamString.\"PatientBirthDay\">\s*\{\s*\")(.+)(\"\s*\}\n)"
    patient_gender = r"(<ParamLong.\"l?PatientSex\">\s*\{\s*)(\d+)(\s*\}\n)"
    patient_age = r"(<ParamDouble.\"flPatientAge\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)"
    patient_weight = r"(<ParamDouble.\"flUsedPatientWeight\">\s*\{\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)"
    patient_height = r"(<ParamDouble.\"flPatientHeight\">\s*\{\s*<Unit> \"\[mm\]\"\s*<Precision> \d+\s*)(\d+\.\d*)(\s*\}\n)"

    header_string = re.sub(patient_name, lambda match: "".join(
        (match.group(1), ("x" * (len(match.group(2)))), match.group(3))),
        header_string)

    header_string = re.sub(patient_id, lambda match: "".join(
        (match.group(1), ("x" * (len(match.group(2)))), match.group(3))),
        header_string)

    header_string = re.sub(patient_birthday, lambda match: "".join(
        (match.group(1), "19700101", match.group(3))),
        header_string)

    header_string = re.sub(patient_gender, lambda match: "".join(
        (match.group(1), "0", match.group(3))),
        header_string)

    header_string = re.sub(patient_age, lambda match: "".join(
        (match.group(1), re.sub(r"\d", "0", match.group(2)), match.group(3))),
        header_string)

    header_string = re.sub(patient_weight, lambda match: "".join(
        (match.group(1), re.sub(r"\d", "0", match.group(2)), match.group(3))),
        header_string)

    header_string = re.sub(patient_height, lambda match: "".join(
        (match.group(1), re.sub(r"\d", "0", match.group(2)), match.group(3))),
        header_string)

    # We need to remove information which contains the date and time of the exam
    # this is not stored in a helpful way which complicates finding it.
    # I think that this FrameOfReference parameter is the correct time, it is
    # certainly the correct date.
    # As Siemens uses date and time to refer to other scans, we need to censor
    # any string which contains the date of this exam. Also some references
    # seem to use the date with the short form of the year so we match that
    frame_of_reference = re.search(r"(<ParamString.\"FrameOfReference\">  { )(\".+\")(  }\n)", header_string).group(2)
    exam_date_time = frame_of_reference.split(".")[10]
    exam_date = exam_date_time[2:8]

    # any string which contains the exam date has all alpha-numerics replaced
    # by the character x
    header_string = re.sub(r"\"[\d\.]*{0}[\d\.]*\"".format(exam_date),
                           lambda match: re.sub(r"\w", "x", match.group()),
                           header_string)

    return header_string


def anonymize_twix_vd(fin, fout):
    twix_id, num_measurements = struct.unpack("II", fin.read(8))

    fout.write(struct.pack("II", twix_id, num_measurements))

    for i in range(num_measurements):
        fin.seek(8 + 152 * i)

        print(fin.tell())
        meas_id, file_id, offset, length, patient_name, protocol_name = struct.unpack("IIQQ64s64s", fin.read(152))
        print(patient_name)
        anon_patient_name = ("x" * 64).encode("latin-1")
        print(anon_patient_name)
        print(offset)
        print(length)
        print(offset + length)
        fout.seek(8 + 152 * i)
        fout.write(struct.pack("IIQQ64s64s", meas_id, file_id, offset, length, anon_patient_name, protocol_name))

        fin.seek(offset)
        # read the header and anonymize it
        header_size = struct.unpack("I", fin.read(4))[0]
        print("header_size: {}".format(header_size))
        header = fin.read(header_size - 4)
        header_string = header[:-24].decode('latin-1')

        anonymized_header = anonymize_twix_header(header_string)
        print("fout position")
        print(fout.tell())
        fout.seek(offset)
        print(fout.tell())
        fout.write(struct.pack("I", header_size))
        fout.write(anonymized_header.encode("latin1"))
        fout.write(header[-24:])

        fout.write(fin.read(length - header_size))


def anonymize_twix_vb(fin, fout):
    # first four bytes are the size of the header
    header_size = struct.unpack("I", fin.read(4))[0]

    # read the rest of the header minus the four bytes we already read
    header = fin.read(header_size - 4)
    # for some reason the last 24 bytes of the header contain some stuff that
    # is not a string, I don't know what it is
    header_string = header[:-24].decode('latin-1')

    anonymized_header = anonymize_twix_header(header_string)

    fout.write(struct.pack("I", header_size))
    fout.write(anonymized_header.encode('latin-1'))
    fout.write(header[-24:])
    fout.write(fin.read())


def anonymize_twix(filename, anonymized_filename):
    with open(filename, 'rb') as fin:

        # we can tell the type of file from the first two uints in the header
        first_uint, second_uint = struct.unpack("II", fin.read(8))

        # reset the file pointer before giving to specific function
        fin.seek(0)

        with open(anonymized_filename, 'wb') as fout:
            if first_uint == 0 and second_uint <= 64:
                anonymize_twix_vd(fin, fout)
            else:
                anonymize_twix_vb(fin, fout)


def get_header(filename):
    with open(filename, 'rb') as fin:

        # we can tell the type of file from the first two uints in the header
        first_uint, second_uint = struct.unpack("II", fin.read(8))

        # reset the file pointer before giving to specific function
        fin.seek(0)


        if first_uint == 0 and second_uint <= 64:
            print("twix file is VD/VE")
            headers = []
            for i in range(second_uint):
                fin.seek(8 + 152 * i)

                meas_id, file_id, offset, length, patient_name, protocol_name = struct.unpack("IIQQ64s64s",
                                                                                              fin.read(152))
                fin.seek(offset)
                # read the header and anonymize it
                header_size = struct.unpack("I", fin.read(4))[0]
                header = fin.read(header_size - 4)
                header_string = header[:-24].decode('latin-1')
                headers.append(header_string)
            return headers

        else:
            print("twix file is VB")
            # read the rest of the header minus the four bytes we already read
            fin.seek(4)
            header = fin.read(first_uint - 4)
            # for some reason the last 24 bytes of the header contain some stuff that
            # is not a string, I don't know what it is
            header_string = header[:-24].decode('latin-1')
            return header_string
