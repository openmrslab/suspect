from suspect import basis

import numpy


def test_gaussian():
    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 10. / 1024 * 2000)
    spectrum = numpy.fft.fft(fid)
    numpy.testing.assert_almost_equal(numpy.amax(spectrum), 0.048099188669422144)
    numpy.testing.assert_allclose(spectrum[5].real, 0.5 * numpy.amax(spectrum), rtol=0.05)
    numpy.testing.assert_allclose(numpy.sum(spectrum), 0.512)


def test_lorentzian():
    time_axis = numpy.arange(0, 0.512, 5e-5)
    fid = basis.lorentzian(time_axis, 0, 0, 10. / 1024 * 2000)
    spectrum = numpy.fft.fft(fid)
    numpy.testing.assert_allclose(numpy.amax(spectrum), 0.031879841710509643)
    numpy.testing.assert_allclose(spectrum[5].real, 0.5 * numpy.amax(spectrum), rtol=0.05)