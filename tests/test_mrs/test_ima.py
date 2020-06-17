import pytest
import numpy.testing
import numpy

import suspect.io.siemens


def test_svs_30():
    data = suspect.io.siemens.load_siemens_dicom("tests/test_data/siemens/SVS_30.IMA")
    assert data.shape == (1024,)
    assert data.te == 30
    assert data.tr == 2000


# def test_svs_30():
#     data = suspect.io.siemens.load_siemens_dicom("suspect/tests/test_data/siemens_svs_30.IMA")
#     assert data.shape == (1024,)
#     numpy.testing.assert_allclose(data.dt, 1/1200, rtol=1e-3)
#     assert data.te == 30


