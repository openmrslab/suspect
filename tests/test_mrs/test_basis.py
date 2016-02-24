from suspect import basis

import numpy


def test_gaussian():
    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 10. / 1024 * 2000)
    spectrum = numpy.fft.fft(fid)
    numpy.testing.assert_almost_equal(numpy.amax(spectrum), 1)
    numpy.testing.assert_almost_equal(spectrum[5].real, 0.5, decimal=1)


def test_lorentzian():
    time_axis = numpy.arange(0, 0.512, 5e-5)
    fid = basis.lorentzian(time_axis, 0, 0, 10. / 1024 * 2000)
    spectrum = numpy.fft.fft(fid)
    numpy.testing.assert_almost_equal(numpy.amax(spectrum), 1)
    numpy.testing.assert_almost_equal(spectrum[5].real, 0.5, decimal=2)