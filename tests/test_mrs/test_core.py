import suspect

import numpy
import pytest


def test_adjust_zero_phase():
    data = suspect.MRSData(numpy.ones(10), 1e-3, 123)
    phased_data = suspect.adjust_phase(data, numpy.pi / 2)
    numpy.testing.assert_allclose(phased_data, 1j * numpy.ones(10))


def test_adjust_first_phase():
    data = suspect.MRSData(numpy.fft.ifft(numpy.ones(10)), 1e-1, 123)
    phased_data = suspect.adjust_phase(data, 0, numpy.pi / 10).spectrum()
    numpy.testing.assert_almost_equal(phased_data[0], -1j)
    numpy.testing.assert_almost_equal(phased_data[-1], numpy.exp(1j * 0.4 * numpy.pi))
    numpy.testing.assert_almost_equal(phased_data[5], 1)


def test_slice_hz():
    data = suspect.MRSData(numpy.ones(1024), 1e-3, 123)
    spectrum = data.spectrum()
    whole_slice = spectrum.slice_hz(-500, 500)
    assert whole_slice == slice(0, 1024)
    restricted_slice = spectrum.slice_hz(-100, 200)
    assert restricted_slice == slice(409, 717)
    with pytest.raises(ValueError):
        too_large_slice = spectrum.slice_hz(-1000, 1000)


def test_slice_ppm():
    data = suspect.MRSData(numpy.ones(1000), 1e-3, 123)
    spectrum = data.spectrum()
    a_slice = spectrum.slice_ppm(5.7, 3.7)
    assert a_slice == slice(377, 623)
    reversed_slice = spectrum.slice_ppm(3.7, 5.7)
    assert a_slice == slice(377, 623)
