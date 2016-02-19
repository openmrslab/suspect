import pytest

import suspect.io.rda


def test_twix_nofile():
    with pytest.raises(FileNotFoundError):
        suspect.io.rda.load_rda("")


def test_csi_file():
    data = suspect.io.rda.load_rda("suspect/tests/test_data/CSITEST_20151028_97_1.rda")
    assert data.shape == (1, 16, 16, 1024)
    assert data.dt == 8.33e-4
    assert data.te == 97

