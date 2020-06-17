import pytest

import suspect.io.rda

import numpy


def test_twix_nofile():
    with pytest.raises(FileNotFoundError):
        suspect.io.rda.load_rda("")


def test_svs_file():
    data = suspect.io.load_rda("tests/test_data/siemens/SVS_30.rda")
    assert data.shape == (1024,)
    assert data.tr == 2000
    assert data.te == 30

    # transform of centre of voxel is same as position
    voxel_position = data.to_scanner(0, 0, 0)
    numpy.testing.assert_allclose(data.metadata["position"], voxel_position)
    numpy.testing.assert_equal(data.position, voxel_position)

    # transform of corner of voxel is same as PositionVector parameter in file
    position_vector = numpy.array([36.834798, 42.553376, 1.117466])
    numpy.testing.assert_allclose(position_vector, data.to_scanner(-0.5, -0.5, 0.0), rtol=1e-5)

#def test_csi_file():
#    data = suspect.io.rda.load_rda("suspect/tests/test_data/CSITEST_20151028_97_1.rda")
#    assert data.shape == (1, 16, 16, 1024)
#    assert data.dt == 8.33e-4
#    assert data.te == 97

