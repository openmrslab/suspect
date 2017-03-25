import suspect

import numpy
import pytest


def test_create_mrs():
    data = suspect.MRSData(numpy.zeros(1024), 5e-4, 123)
    assert data.shape == (1024,)
    assert data.dt == 5e-4
    assert data.np == 1024
    assert data.df == 1.953125


def test_slice_mrs():
    data = suspect.MRSData(numpy.ones((2, 1024), 'complex'), 5e-4, 123)
    repetition = data[0]
    assert repetition.shape == (1024,)
    assert repetition.dt == 5e-4


def test_average_mrs():
    data = suspect.MRSData(numpy.ones((2, 1024), 'complex'), 5e-4, 123)
    averaged_data = data.inherit(numpy.average(data, axis=0))
    assert type(averaged_data) == suspect.MRSData
    assert averaged_data.dt == 5e-4
    assert averaged_data.f0 == 123


def test_zero_rank_to_scalar():
    data = suspect.MRSData(numpy.ones(1024, 'complex'), 5e-4, 123)
    sum = numpy.sum(data)
    assert numpy.isscalar(sum)


def test_spectrum_2_fid():
    data = suspect.MRSData(numpy.ones(1024, 'complex'), 5e-4, 123)
    spectrum = data.spectrum()
    assert type(spectrum) == suspect.MRSSpectrum
    numpy.testing.assert_equal(spectrum, numpy.fft.fftshift(numpy.fft.fft(data)))
    fid = spectrum.fid()
    numpy.testing.assert_equal(data, fid)
