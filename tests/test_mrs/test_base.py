import suspect
import numpy as np
import pytest
import suspect.base
from suspect import _transforms


def test_create_base():
    base = suspect.base.ImageBase(np.zeros(1), np.eye(4))
    np.testing.assert_equal(base.transform, np.eye(4))
    vec = (1, 0, 0)
    tvec = base.to_scanner(*vec)
    np.testing.assert_equal(np.array(vec), tvec)


def test_base_transform():
    position = np.array([10, 20, 30])
    voxel_size = np.array([20, 20, 20])
    transform = suspect.transformation_matrix([1, 0, 0], [0, 1, 0], position, voxel_size)
    base = suspect.base.ImageBase(np.zeros(1), transform)
    np.testing.assert_equal(base.position, position)
    np.testing.assert_equal(base.voxel_size, voxel_size)
    transformed = base.to_scanner(0, 0, 0)
    np.testing.assert_equal(transformed, position)
    transformed = base.to_scanner(np.array([[0, 0, 0], [1, 1, 1]]))
    np.testing.assert_equal(transformed, [position, position + voxel_size])
    transformed = base.from_scanner(position)
    np.testing.assert_equal((0, 0, 0), transformed)


def test_transforms_fail():
    base = suspect.base.ImageBase(np.zeros(1))
    with pytest.raises(ValueError):
        base.to_scanner(0, 0, 0)
    with pytest.raises(ValueError):
        base.from_scanner(0, 0, 0)
    with pytest.raises(ValueError):
        pos = base.position
    with pytest.raises(ValueError):
        vox = base.voxel_size
    with pytest.raises(ValueError):
        _transforms.normalise_positions_for_transform(0, 0)
    with pytest.raises(ValueError):
        slice_vec = base.slice_vector
    with pytest.raises(ValueError):
        slice_vec = base.row_vector
    with pytest.raises(ValueError):
        slice_vec = base.col_vector


def test_find_axes():
    transform = _transforms.transformation_matrix([2, 1, 0],
                                                  [-1, 2, 0],
                                                  [0, 0, 0],
                                                  [1, 1, 1])
    base = suspect.base.ImageBase(np.zeros(1), transform=transform)
    np.testing.assert_equal(base.axial_vector, base.slice_vector)
    np.testing.assert_equal(base.coronal_vector, base.col_vector)
    np.testing.assert_equal(base.sagittal_vector, base.row_vector)


def test_find_axes_reversed():
    transform = _transforms.transformation_matrix([-2, -1, 0],
                                                  [1, -2, 0],
                                                  [0, 0, 0],
                                                  [1, 1, 1])
    base = suspect.base.ImageBase(np.zeros(1), transform=transform)
    np.testing.assert_equal(base.axial_vector, base.slice_vector)
    np.testing.assert_equal(base.coronal_vector, -base.col_vector)
    np.testing.assert_equal(base.sagittal_vector, -base.row_vector)


def test_find_axes_oblong():
    transform = _transforms.transformation_matrix([2, 1, 0],
                                                  [-1, 2, 0],
                                                  [0, 0, 0],
                                                  [40, 4, 1])
    base = suspect.base.ImageBase(np.zeros(1), transform=transform)
    np.testing.assert_equal(base.axial_vector, base.slice_vector)
    np.testing.assert_equal(base.coronal_vector, base.col_vector)
    np.testing.assert_equal(base.sagittal_vector, base.row_vector)


def test_centre():
    transform = _transforms.transformation_matrix([1, 0, 0],
                                                  [0, 1, 0],
                                                  [-127.5, -99.5, -99],
                                                  [1, 1, 2])
    base = suspect.base.ImageBase(np.zeros((128, 256, 256)), transform=transform)
    np.testing.assert_equal(base.centre, [0, 28, 28])


def test_resample():
    transform = _transforms.transformation_matrix([1, 0, 0],
                                                  [0, 1, 0],
                                                  [-1.5, -0.5, -1.5],
                                                  [1, 1, 1])
    data = np.random.rand(4, 2, 4)
    base = suspect.base.ImageBase(data, transform=transform)
    resampled_1 = base.resample([1, 0, 0], [0, 1, 0], [1, 2, 4],
                                centre=[0, 0, 0.5])
    np.testing.assert_equal(base[2], resampled_1)
    resampled_1_transform = np.array([[1, 0, 0, -1.5],
                                      [0, 1, 0, -0.5],
                                      [0, 0, 1, 0.5],
                                      [0, 0, 0, 1]])
    np.testing.assert_equal(resampled_1_transform, resampled_1.transform)
    resampled_2 = base.resample([0, 1, 0], [0, 0, 2], [1, 4, 2],
                                centre=[0.5, 0, 0])
    np.testing.assert_equal(base[:, :, 2], resampled_2)
    resampled_3 = base.resample([1, 0, 0], [0, 0, 1], [1, 4, 4],
                                centre=[0, 0.5, 0])
    np.testing.assert_equal(base[:, 1], resampled_3)
