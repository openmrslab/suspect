import suspect
import numpy
import pytest
import suspect.base
from suspect import _transforms


def test_create_base():
    base = suspect.base.ImageBase(numpy.zeros(1), numpy.eye(4))
    numpy.testing.assert_equal(base.transform, numpy.eye(4))
    vec = (1, 0, 0)
    tvec = base.to_scanner(*vec)
    numpy.testing.assert_equal(numpy.array(vec), tvec)


def test_base_transform():
    position = numpy.array([10, 20, 30])
    voxel_size = numpy.array([20, 20, 20])
    transform = suspect.transformation_matrix([1, 0, 0], [0, 1, 0], position, voxel_size)
    base = suspect.base.ImageBase(numpy.zeros(1), transform)
    numpy.testing.assert_equal(base.position, position)
    numpy.testing.assert_equal(base.voxel_size, voxel_size)
    transformed = base.to_scanner(0, 0, 0)
    numpy.testing.assert_equal(transformed, position)
    transformed = base.to_scanner(numpy.array([[0, 0, 0], [1, 1, 1]]))
    numpy.testing.assert_equal(transformed, [position, position + voxel_size])
    transformed = base.from_scanner(position)
    numpy.testing.assert_equal((0, 0, 0), transformed)


def test_transforms_fail():
    base = suspect.base.ImageBase(numpy.zeros(1))
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
