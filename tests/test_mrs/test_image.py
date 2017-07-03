import suspect

import suspect._transforms

import numpy as np


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
