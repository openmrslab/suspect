import pytest
import numpy

import suspect.io.twix


def test_twix_nofile():
    with pytest.raises(FileNotFoundError):
        suspect.io.twix.load_twix("")


def test_veriofile():
    data = suspect.io.load_twix("tests/test_data/siemens/twix_vb.dat")
    assert data.shape == (128, 32, 2048)
    assert data.np == 2048
    assert data.dt == 2.5e-4
    assert data.te == 30.0
    assert data.tr == 2000
    numpy.testing.assert_almost_equal(data.f0, 123.261716)
    numpy.testing.assert_allclose(data.transform, numpy.array(
        [[-20, 0, 0, 4.917676],
         [0, 20, 0, 57.525424],
         [0, 0, -20, 43.220339],
         [0, 0, 0, 1]]
    ))


#def test_skyra():
#    data = suspect.io.load_twix("tests/test_data/twix_vd_csi.dat")
#    assert data.np == 2048


#def test_anonymize_verio():
#    data = suspect.io.load_twix("suspect/tests/test_data/meas_MID178_svs_se_30_PCG_FID95017.dat")
#    suspect.io.twix.anonymize_twix("suspect/tests/test_data/meas_MID178_svs_se_30_PCG_FID95017.dat", "suspect/tests/test_data/twix_vb.dat")
#    data = suspect.io.load_twix("suspect/tests/test_data/twix_vb.dat")
#    assert data.metadata["patient_name"] == "x" * 13
#    assert data.metadata["patient_id"] == "x" * 6
#    assert data.metadata["patient_birthdate"] == "19700101"
