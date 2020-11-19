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

    # Test with TWIX VD data (scanned on VIDA scanner)
    data = suspect.io.load_twix("tests/test_data/siemens/twix_vd.dat")
    assert data.shape == (64, 2, 9, 2048)
    assert data.np == 2048
    assert data.dt == 3.333e-4
    assert data.te == 135.0
    assert data.tr == 2000
    numpy.testing.assert_almost_equal(data.f0, 123.256306)
    numpy.testing.assert_allclose(data.transform, numpy.array(
        [[-20, 0, 0, 38.341346],
         [0, 20, 0, -2.531308],
         [0, 0, -20, 6.560502],
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

def test_calculate_orientation():
    assert suspect.io.twix.calculate_orientation([1, 1, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([0.5, 0.5, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([1, 1, 0.5]) == "COR"
    assert suspect.io.twix.calculate_orientation([0.5, 1, 0.5]) == "COR"
    assert suspect.io.twix.calculate_orientation([1, 0.5, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([1, 0.5, 0.5]) == "SAG"
    assert suspect.io.twix.calculate_orientation([0.5, 1, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([1, 0.5, 0.5]) == "SAG"
    assert suspect.io.twix.calculate_orientation([0.5, 0, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([0, 1, 0.5]) == "COR"
    assert suspect.io.twix.calculate_orientation([0, 0.5, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([0.5, 1, 0]) == "COR"
    assert suspect.io.twix.calculate_orientation([1, 0, 0]) == "SAG"
    assert suspect.io.twix.calculate_orientation([0, 1, 0.5]) == "COR"
    assert suspect.io.twix.calculate_orientation([0.5, 0.5, 1]) == "TRA"
    assert suspect.io.twix.calculate_orientation([1, 0.5, 0]) == "SAG"
    assert suspect.io.twix.calculate_orientation([0, 1, 0]) == "COR"
    assert suspect.io.twix.calculate_orientation([1, 0, 0.5]) == "SAG"
    assert suspect.io.twix.calculate_orientation([0, 0, 1]) == "TRA"
