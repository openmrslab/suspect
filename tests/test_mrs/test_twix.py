import suspect.mrs.twix

import pytest
import numpy
import numpy.testing


def test_twix_nofile():
    with pytest.raises(FileNotFoundError):
        suspect.mrs.twix.load_twix("")


#def test_veriofile():
#    data = suspect.mrs.twix.load_twix("suspect/tests/test_data/meas_MID178_svs_se_30_PCG_FID95017.dat")
#    assert data.shape == (128, 32, 2048)
#    assert data.np == 2048
#    assert data.dt == 2.5e-4
#    numpy.testing.assert_almost_equal(data.f0, 123.261716)

if __name__ == "__main__":
    #test_veriofile()
    pass