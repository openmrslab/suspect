import suspect

import nibabel
import numpy
import os
import pydicom
import pydicom.errors


def load_dicom_volume(filename):
    """ Creates a 3D volume from all the slices in a folder and extracts useful
    information from a supplied image. The function will attempt to read all
    files in the folder which have the same extension as the supplied filename
    and combine all slices with a matching SeriesInstanceUID.

    Parameters
    ----------
    filename : DICOM file

    Returns
    -------
    dict
        A dictionary containing values for voxel spacing, position, volume, vectors, and a transformation matrix

    """
    _, file_ext = os.path.splitext(filename)
    # load the supplied file and get the UID of the series
    ds = pydicom.read_file(filename)
    seriesUID = ds.SeriesInstanceUID

    # get the position of the image
    position = numpy.array(list(map(float, ds.ImagePositionPatient)))

    # get the direction normal to the plane of the image
    row_vector = numpy.array(ds.ImageOrientationPatient[:3])
    col_vector = numpy.array(ds.ImageOrientationPatient[3:])
    normal_vector = numpy.cross(row_vector, col_vector)

    # we order slices by their distance along the normal
    def normal_distance(coords):
        return numpy.dot(normal_vector, coords)

    # create a dictionary to hold the slices as we load them
    slices = {normal_distance(position): ds.pixel_array}

    # extract the path to the folder of the file so we can look for others from the same series
    folder, _ = os.path.split(filename)
    for name in os.listdir(folder):
        if name.endswith(file_ext): # name.lower().endswith(".ima") or name.lower().endswith(".dcm"):
            new_dicom_name = os.path.join(folder, name)
            try:
                new_ds = pydicom.read_file(new_dicom_name)
            except pydicom.errors.InvalidDicomError as e:
                continue

            # check that the series UID matches
            if new_ds.SeriesInstanceUID == seriesUID:
                if new_ds.pixel_array.shape != ds.pixel_array.shape:
                    continue
                new_position = list(map(float, new_ds.ImagePositionPatient))
                slices[normal_distance(new_position)] = new_ds.pixel_array

                # we set the overall position of the volume with the position
                # of the lowest slice
                if normal_distance(new_position) < normal_distance(position):
                    position = new_position

    # that is all the slices in the folder, assemble them into a 3d volume
    voxel_array = numpy.zeros((len(slices),
                               ds.pixel_array.shape[0],
                               ds.pixel_array.shape[1]), dtype=ds.pixel_array.dtype)
    sorted_slice_positions = sorted(slices.keys())
    for i, slice_position in enumerate(sorted_slice_positions):
        voxel_array[i] = slices[slice_position]

    # the voxel spacing is a combination of PixelSpacing and slice separation
    voxel_spacing = list(map(float, ds.PixelSpacing))
    voxel_spacing.append(sorted_slice_positions[1] - sorted_slice_positions[0])

    # replace the initial slice z position with the lowest slice z position
    # position[2] = sorted_slice_positions[0]

    transform = suspect.transformation_matrix(row_vector,
                                              col_vector,
                                              position,
                                              voxel_spacing)

    return suspect.base.ImageBase(voxel_array, transform)


def load_nifti(filename):
    """
    Load the 3D volume contained in the supplied Nifti file, and convert the
    coordinate system to the DICOM standard. This means transposing the data
    to use slices as the first index and columns as the last, and negating
    the x and y axes.
    
    Parameters
    ----------
    filename : str
        The filename from which to load the data
        
    Returns
    -------
    suspect.base.ImageBase
        3D image volume
    """
    # start by loading the nifti file using nibabel
    nii = nibabel.load(filename)

    # nibabel loads cols, rows, slices, so we transpose to match DICOM
    image = suspect.base.ImageBase(nii.get_fdata().T, nii.affine)

    # nifti also uses a reversed coordinate system for x and y, so negate them
    image.transform[:2] *= -1.0

    return image


def save_nifti(filename, image):
    """
    Save a 3D volume to a file in the Nifti1 format.
    
    Parameters
    ----------
    filename : str
        The filename where the file should be saved
    image : suspect.base.ImageBase
        The volume to save
    """
    # we have to modify the transform to use nifti coordinates (negate x and y)
    affine = image.transform.copy()
    affine[:2] *= -1.0

    nii = nibabel.nifti1.Nifti1Image(image.T, affine)
    nii.to_filename(filename)
