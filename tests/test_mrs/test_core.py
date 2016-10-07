import suspect

import numpy


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
