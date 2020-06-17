from suspect import MRSData, transformation_matrix

import functools
import numpy

try:
    import GERecon
except ModuleNotFoundError as e:
    GERecon = e


gerecon_err_msg = "GE P-file functionality requires the GE \"Orchestra for " \
  "Python\" package to be installed. For more information, please visit " \
  "http://suspect.readthedocs.io/en/latest/pfiles.html"


def requires_gerecon(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if isinstance(GERecon, ModuleNotFoundError):
            raise ModuleNotFoundError(gerecon_err_msg)
        return func(self, *args, **kwargs)

    return wrapper


def extract_header_parameters(header):
    """
    Creates a dictionary with the most important header parameters from a GE
    P-file, used to create an MRSData object with the result.

    Parameters
    ----------
    header: dict
        The header dictionary loaded by the Orchestra library from a P-file

    Returns
    -------
    dict
        Dictionary containing the parameters needed to create an MRSData object
    """
    dt = 1 / header["rdb_hdr_rec"]["spectral_width"]
    f0 = header["rdb_hdr_ps"]["mps_freq"] * 1e-7
    te = header["rdb_hdr_rec"]["rdb_hdr_te"] / 1000  # convert from us to ms
    tr = header["rdb_hdr_image"]["tr"] / 1000  # convert from us to ms

    # calculating the transform is quite involved
    # GE internally uses a RAS coordinate system, so we have to convert to the
    # LPS system used here (and by DICOM standard)
    voxel_size = numpy.array([header["rdb_hdr_image"]["user8"],
                              header["rdb_hdr_image"]["user9"],
                              header["rdb_hdr_image"]["user10"]])
    position_vector = numpy.array([-header["rdb_hdr_image"]["user11"],
                                   -header["rdb_hdr_image"]["user12"],
                                   header["rdb_hdr_image"]["user13"]])
    tl_coord_lps = numpy.array([-header["rdb_hdr_image"]["tlhc_R"],
                                -header["rdb_hdr_image"]["tlhc_A"],
                                header["rdb_hdr_image"]["tlhc_S"]])

    tr_coord_lps = numpy.array([-header["rdb_hdr_image"]["trhc_R"],
                                -header["rdb_hdr_image"]["trhc_A"],
                                header["rdb_hdr_image"]["trhc_S"]])

    br_coord_lps = numpy.array([-header["rdb_hdr_image"]["brhc_R"],
                                -header["rdb_hdr_image"]["brhc_A"],
                                header["rdb_hdr_image"]["brhc_S"]])

    e1 = tr_coord_lps - tl_coord_lps
    e1 = e1 / numpy.linalg.norm(e1)
    e2 = br_coord_lps - tr_coord_lps
    e2 = e2 / numpy.linalg.norm(e2)

    transform = transformation_matrix(e1,
                                      e2,
                                      position_vector,
                                      voxel_size)

    return {
        "dt": dt,
        "f0": f0,
        "te": te,
        "tr": tr,
        "transform": transform
    }


@requires_gerecon
def load_pfile(filename):
    """
    Loads MRS data from a GE P-file.

    Parameters
    ----------
    filename: str
        The filename of the P-file to be loaded.

    Returns
    -------
    MRSData
        The loaded FID data in a Suspect MRSData object
    """
    pfile = GERecon.Pfile(filename)

    header = pfile.Header()

    # check if this is a single voxel, CSI or something else
    # the check for a single voxel is that the xcsi, ycsi and zcsi parameters
    # are all 1
    # the check for a CSI acquisition is that the number of points in an FID
    # as contained in the user1 parameter matches up to the x_res metadata
    # parameter (and that it is not a single voxel)
    # otherwise we assume a more complicated readout (spiral/epsi etc.) and
    # just return the data as it comes
    if header["rdb_hdr_rec"]["xcsi"] == 1 and \
       header["rdb_hdr_rec"]["ycsi"] == 1 and \
       header["rdb_hdr_rec"]["zcsi"] == 1:
        return prepare_pfile_svs(pfile)
    elif header["rdb_hdr_rec"]["rdb_hdr_user1"] == pfile.MetaData()["acquiredXRes"]:
        return prepare_pfile_csi(pfile)
    else:
        return prepare_pfile_advanced(pfile)


def prepare_pfile_svs(pfile):
    """
    Transform a P-file containing SVS MRS data into an MRSData object.

    Single voxel spectroscopy on GE is the most complicated case to process.

    Parameters
    ----------
    pfile
        P-file loaded with GE's Orchestra for Python library.

    Returns
    -------
        The raw FID data loaded into a Suspect MRSData object.
    """
    header = pfile.Header()
    metadata = pfile.MetaData()

    num_echoes = int(header["rdb_hdr_rec"]["rdb_hdr_nechoes"])
    num_frames = int(header["rdb_hdr_rec"]["rdb_hdr_nframes"])
    num_averages = int(header["rdb_hdr_rec"]["rdb_hdr_navs"])

    # read the number of (water-suppressed) spectra acquired
    data_frames = int(header["rdb_hdr_rec"]["rdb_hdr_user4"])
    # in some cases GE automatically combines blocks of num_averages on the
    # scanner depending on the value of the no_add parameter
    # I think that this is stored in the lsb of the below header parameter
    no_add = header["rdb_hdr_image"]["user24"] % 2 == 1
    if no_add is not True:
        data_frames //= num_averages
    # the header parameter with the number of reference frames is not reliable:
    # sometimes it ignores the value of no_add so we just assume that all non
    # data frames are ref frames
    ref_frames = num_frames - data_frames

    # when we load the data from the pfile, we do it echo by echo
    # the order of the other axes is initially what is returned from pfile
    # we will reshape and reorder it afterwards
    raw_data = numpy.zeros((num_echoes,
                            metadata["acquiredXRes"],
                            metadata["acquiredYRes"],
                            metadata["channels"]), dtype=numpy.complex)

    for i in range(num_echoes):
        raw_data[i] = pfile.KSpace(0, i)

    # if no_add was set, every other average will have opposite sign and has to
    # be flipped
    if no_add:
        # TODO we assume that the flipping happens within a set of num_averages
        # TODO and not outside, should check this is the case
        flip_array = numpy.array(-1) ** numpy.arange(num_averages)
        flip_array = numpy.tile(flip_array, int(num_frames / num_averages))
        flip_array = flip_array[numpy.newaxis, numpy.newaxis, :, numpy.newaxis]
        raw_data *= flip_array

    # reorganise the data to be echoes, averages, channels and ADC
    raw_data = raw_data.transpose((0, 2, 3, 1))

    # we have the acquired data, now put together the desired metadata
    header_params = extract_header_parameters(header)

    mrs_data = MRSData(raw_data,
                       **header_params)

    print(ref_frames)
    print(metadata)
    print(mrs_data.shape)

    wref = mrs_data[:, :ref_frames].reshape(-1, metadata["channels"], metadata["acquiredXRes"]).squeeze()
    data = mrs_data[:, ref_frames:].squeeze()

    return data, wref


def prepare_pfile_csi(pfile):
    """
    Transform a P-file containing CSI MRS data into an MRSData object.

    The GE CSI sequence seems to be very straightforward, just acquiring a full
    grid of phase encoded spectra with no variable averages, and no included
    water reference data. This function simply unpacks the acquired "y-axis"
    from the P-file into the 2 or 3 phase encoded axes of the CSI grid, and
    reorders the axes to the standard Suspect sequence of echoes, phase
    encodes, channels and ADC points.

    Parameters
    ----------
    pfile
        P-file loaded with GE's Orchestra for Python library.

    Returns
    -------
        The raw FID data loaded into a Suspect MRSData object.
    """
    header = pfile.Header()
    metadata = pfile.MetaData()

    num_echoes = int(header["rdb_hdr_rec"]["rdb_hdr_nechoes"])
    x_res = int(header["rdb_hdr_rec"]["xcsi"])
    y_res = int(header["rdb_hdr_rec"]["ycsi"])
    z_res = int(header["rdb_hdr_rec"]["zcsi"])

    # when we load the data from the pfile, we do it echo by echo
    # the order of the other axes is initially what is returned from pfile
    # we will reshape and reorder it afterwards
    raw_data = numpy.zeros((num_echoes,
                            metadata["acquiredXRes"],
                            metadata["acquiredYRes"],
                            metadata["channels"]), dtype=numpy.complex)

    for i in range(num_echoes):
        raw_data[i] = pfile.KSpace(0, i)

    # reorganise the data to be echoes, averages, channels and ADC
    # then squeeze to remove any size 1 axes (e.g. echoes or channels)
    raw_data = raw_data.transpose((0, 2, 3, 1))\
        .reshape((num_echoes,
                  z_res,
                  y_res,
                  x_res,
                  metadata["channels"],
                  metadata["acquiredXRes"]))\
        .squeeze()

    return MRSData(raw_data,
                   **extract_header_parameters(header))


def prepare_pfile_advanced(pfile):
    """
    Transform a P-file containing advanced MRS data into an MRSData object.

    Despite the name, this is the simplest P-file preparation option as it does
    nothing to the data apart from unpacking it into an ndarray and adding the
    necessary Suspect metadata.

    Parameters
    ----------
    pfile
        P-file loaded with GE's Orchestra for Python library.

    Returns
    -------
    MRSData
        The raw FID data loaded into a Suspect MRSData object.
    """
    header = pfile.Header()
    metadata = pfile.MetaData()

    num_echoes = int(header["rdb_hdr_rec"]["rdb_hdr_nechoes"])

    # when we load the data from the pfile, we do it echo by echo
    # the order of the other axes is initially what is returned from pfile
    # we will reshape and reorder it afterwards
    raw_data = numpy.zeros((num_echoes,
                            metadata["acquiredXRes"],
                            metadata["acquiredYRes"],
                            metadata["channels"]), dtype=numpy.complex)

    for i in range(num_echoes):
        raw_data[i] = pfile.KSpace(0, i)

    # reorganise the data to put channels and ADC after the averages/phase encodes
    # then squeeze to remove any size 1 axes (e.g. echoes or channels)
    raw_data = raw_data.transpose((0, 2, 3, 1)).squeeze()

    return MRSData(raw_data,
                   **extract_header_parameters(header))
