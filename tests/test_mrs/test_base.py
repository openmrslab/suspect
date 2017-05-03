import suspect
import numpy
import suspect.base


def test_create_base():
    base = suspect.base.ImageBase(numpy.zeros(1), numpy.eye(4))
    numpy.testing.assert_equal(base.transform, numpy.eye(4))
    vec = (1, 0, 0)
    tvec = base.to_scanner(*vec)
    numpy.testing.assert_equal(numpy.array(vec), tvec)


def test_base_transform():
    position = [10, 20, 30]
    voxel_size = [20, 20, 20]
    transform = suspect.transformation_matrix([1, 0, 0], [0, 1, 0], position, voxel_size)
    base = suspect.base.ImageBase(numpy.zeros(1), transform)
    numpy.testing.assert_equal(base.position, position)
    numpy.testing.assert_equal(base.voxel_size, voxel_size)
