import suspect

import numpy


def test_null_transform():
    fid = numpy.ones(128, 'complex')
    data = suspect.MRSData(fid, 1.0 / 128, 123)
    transformed_data = suspect.processing.frequency_correction.transform_fid(data, 0, 0)
    assert type(transformed_data) == suspect.MRSData


def test_water_peak_alignment():
    spectrum = numpy.zeros(128, 'complex')
    spectrum[0] = 1
    fids = suspect.MRSData(numpy.zeros((16, 128), 'complex'), 1.0 / 128, 123)
    for i in range(fids.shape[0]):
        rolled_spectrum = numpy.roll(spectrum, i)
        fids[i] = numpy.fft.ifft(rolled_spectrum)
        frequency_shift = suspect.processing.frequency_correction.residual_water_alignment(fids[i])
        numpy.testing.assert_almost_equal(frequency_shift, i)


def test_frequency_transform():
    spectrum = numpy.zeros(128, 'complex')
    spectrum[0] = 1

    for i in range(16):
        rolled_spectrum = numpy.roll(spectrum, i)
        fid = suspect.MRSData(numpy.fft.ifft(rolled_spectrum), 1.0 / 128, 123)
        transformed_fid = suspect.processing.frequency_correction.transform_fid(fid, -i, 0)
        transformed_spectrum = numpy.fft.fft(transformed_fid)
        numpy.testing.assert_almost_equal(transformed_spectrum, spectrum)


def test_apodize():
    data = suspect.MRSData(numpy.ones(1024), 5e-4, 123.456)
    apodized_data = suspect.processing.apodize(data, suspect.processing.gaussian_window, {"line_broadening": data.df * 8})
    spectrum = numpy.fft.fft(apodized_data)
    numpy.testing.assert_almost_equal(numpy.amax(spectrum), 1)
    numpy.testing.assert_almost_equal(spectrum[4].real, 0.5, decimal=2)


def test_gaussian_denoising():
    # constant signal denoised should be the same as original
    data = numpy.ones(128)
    denoised_data = suspect.processing.denoising.sliding_gaussian(data, 11)
    numpy.testing.assert_almost_equal(data, denoised_data)