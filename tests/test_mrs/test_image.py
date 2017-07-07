import suspect

import suspect._transforms

import numpy as np
import os


def test_simple_mask():
    source_transform = suspect._transforms.transformation_matrix([1, 0, 0],
                                                                 [0, 1, 0],
                                                                 [5, 0, 0],
                                                                 [10, 10, 10])
    ref_transform = suspect._transforms.transformation_matrix([1, 0, 0],
                                                              [0, 1, 0],
                                                              [-10, -5, -5],
                                                              [1, 1, 1])
    source_volume = suspect.MRSBase(np.ones(1024), 1e-3, 123, transform=source_transform)
    ref_volume = suspect.base.ImageBase(np.zeros((20, 20, 20)), transform=ref_transform)
    mask = suspect.image.create_mask(source_volume, ref_volume)
    assert ref_volume.shape == mask.shape
    mask_target = np.zeros_like(ref_volume)
    mask_target[0:10, 0:10, 10:20] = 1
    np.testing.assert_equal(mask_target.astype('bool'), mask)


def test_nifti_io():
    dicom_volume = suspect.image.load_dicom_volume("tests/test_data/siemens/mri/T1.0001.IMA")
    # save in a temporary nifti file
    os.makedirs("tests/test_data/tmp", exist_ok=True)
    suspect.image.save_nifti("tests/test_data/tmp/nifti.nii", dicom_volume)
    nifti_volume = suspect.image.load_nifti("tests/test_data/tmp/nifti.nii")
    np.testing.assert_equal(dicom_volume, nifti_volume)
    np.testing.assert_allclose(dicom_volume.transform, nifti_volume.transform)
