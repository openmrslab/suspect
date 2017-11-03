import numpy as np

from ..base import ImageBase


def create_mask(source_image, ref_image, voxels=None):
    """
    Creates a volumetric mask for the source_image voxel in the coordinate
    system of the ref_image volume.
    
    Parameters
    ----------
    source_image : MRSBase
        The spectroscopy volume from which to create the mask.
    ref_image : ImageBase
        The reference image volume which defines the coordinate system for
        the mask.
    
    Returns
    -------
    numpy.ndarray
        Boolean array with the same shape as ref_image, True for all voxels
        inside source_image, false for all others.
    """

    # create a grid of coordinates for all points in the ref_image
    # the ref_image has coord index order [z, y, x] so we reverse the shape
    # to get the indices in (x, y, z) format for the coordinate conversion
    # make sure that ref_image has 3 dimensions so that we can transform them
    ref_image = np.atleast_3d(ref_image)
    ref_coords = np.mgrid[[range(0, size) for size in np.atleast_3d(ref_image).shape[::-1]]]

    # mgrid puts the (x, y, z) tuple at the front, we want it at the back
    ref_coords = np.moveaxis(ref_coords, 0, -1)

    # now we can apply to_scanner and from_scanner to convert from ref coords
    # into source coords
    scanner_coords = ref_image.to_scanner(ref_coords)
    source_coords = source_image.from_scanner(scanner_coords)

    # now check whether the source_coords are in the selected voxel
    # TODO for now, we assume single voxel data until issue 50 is resolved

    # have to transpose the result to get it to match the shape of ref_image
    mask_volume = np.all((source_coords[..., 0] < 0.5,
                          source_coords[..., 0] >= -0.5,
                          source_coords[..., 1] >= -0.5,
                          source_coords[..., 2] >= -0.5,
                          source_coords[..., 1] < 0.5,
                          source_coords[..., 2] < 0.5), axis=0).T

    return ImageBase(mask_volume, ref_image.transform)
