import suspect

import numpy
import warnings

warnings.filterwarnings('error')


def test_null_transform():
    fid = numpy.ones(128, 'complex')
    data = suspect.MRSData(fid, 1.0 / 128, 123)
    transformed_data = data.adjust_frequency(0)
    transformed_data = transformed_data.adjust_phase(0, 0)
    assert type(transformed_data) == suspect.MRSData
    numpy.testing.assert_equal(transformed_data, data)
    # test again using suspect namespace
    transformed_data = suspect.adjust_frequency(data, 0)
    transformed_data = suspect.adjust_phase(transformed_data, 0, 0)
    numpy.testing.assert_equal(transformed_data, data)



def test_water_peak_alignment_misshape():
    spectrum = numpy.zeros(128, 'complex')
    spectrum[0] = 1
    fids = suspect.MRSData(numpy.zeros((16, 128), 'complex'), 1.0 / 128, 123)
    for i in range(fids.shape[0]):
        rolled_spectrum = numpy.roll(spectrum, i)
        fids[i] = numpy.fft.ifft(rolled_spectrum)
        current_fid = numpy.reshape(fids[i], (1, 128))
        frequency_shift = suspect.processing.frequency_correction.residual_water_alignment(current_fid)
        numpy.testing.assert_almost_equal(frequency_shift, i)


def test_water_peak_alignment():
    spectrum = numpy.zeros(128, 'complex')
    spectrum[0] = 1
    fids = suspect.MRSData(numpy.zeros((16, 128), 'complex'), 1.0 / 128, 123)
    for i in range(fids.shape[0]):
        rolled_spectrum = numpy.roll(spectrum, i)
        fids[i] = numpy.fft.ifft(rolled_spectrum)
        frequency_shift = suspect.processing.frequency_correction.residual_water_alignment(fids[i])
        numpy.testing.assert_almost_equal(frequency_shift, i)


def test_spectral_registration():
    time_axis = numpy.arange(0, 0.512, 5e-4)
    target_fid = suspect.MRSData(suspect.basis.gaussian(time_axis, 0, 0, 50.0), 5e-4, 123)
    for i in range(1, 15):
        input_fid = suspect.MRSData(suspect.basis.gaussian(time_axis, i, 0, 50.0), 5e-4, 123)
        frequency_shift, phase_shift = suspect.processing.frequency_correction.spectral_registration(input_fid, target_fid)
        numpy.testing.assert_allclose(frequency_shift, i)


def test_compare_frequency_correction():
    test_data = suspect.io.load_twix("tests/test_data/siemens/twix_vb.dat")
    test_data = test_data.inherit(numpy.average(test_data, axis=1, weights=suspect.processing.channel_combination.svd_weighting(numpy.average(test_data, axis=0))))
    sr_target = test_data[0]
    for i in range(test_data.shape[0]):
        current_fid = test_data[i]
        wpa_fs = suspect.processing.frequency_correction.residual_water_alignment(current_fid)
        sr_fs = suspect.processing.frequency_correction.spectral_registration(current_fid, sr_target)[0]
        numpy.testing.assert_allclose(wpa_fs, sr_fs, atol=current_fid.df)


def test_frequency_transform():
    spectrum = numpy.zeros(128, 'complex')
    spectrum[0] = 1

    for i in range(16):
        rolled_spectrum = numpy.roll(spectrum, i)
        fid = suspect.MRSData(numpy.fft.ifft(rolled_spectrum), 1.0 / 128, 123)
        transformed_fid = fid.adjust_frequency(-i)
        transformed_spectrum = numpy.fft.fft(transformed_fid)
        numpy.testing.assert_almost_equal(transformed_spectrum, spectrum)


def test_apodize():
    data = suspect.MRSData(numpy.ones(1024), 5e-4, 123.456)
    raw_spectrum = numpy.fft.fft(data)
    apodized_data = suspect.processing.apodize(data, suspect.processing.gaussian_window, {"line_broadening": data.df * 8})
    spectrum = numpy.fft.fft(apodized_data)
    numpy.testing.assert_allclose(spectrum[4].real, 0.5 * numpy.amax(spectrum), rtol=0.01)
    numpy.testing.assert_allclose(numpy.sum(spectrum), numpy.sum(raw_spectrum))


def test_gaussian_denoising():
    # constant signal denoised should be the same as original
    data = numpy.ones(128)
    denoised_data = suspect.processing.denoising.sliding_gaussian(data, 11)
    numpy.testing.assert_almost_equal(data, denoised_data)


def test_svd_dtype():
    data = numpy.ones(128, dtype=complex)
    denoised_data = suspect.processing.denoising.svd(data, 8)
    assert data.dtype == denoised_data.dtype


def test_sliding_window_dtype():
    data = numpy.ones(128, dtype=complex)
    denoised_data = suspect.processing.denoising.sliding_window(data, 30)
    assert data.dtype == denoised_data.dtype


def test_sliding_gaussian_dtype():
    data = numpy.ones(128, dtype=complex)
    denoised_data = suspect.processing.denoising.sliding_gaussian(data, 30)
    assert data.dtype == denoised_data.dtype


def test_water_suppression():
    data = suspect.io.load_twix("tests/test_data/siemens/twix_vb.dat")
    channel_combined_data = data.inherit(numpy.average(data, axis=1))
    components = suspect.processing.water_suppression.hsvd(channel_combined_data[10], 4, int(data.np / 2))
    fid = suspect.processing.water_suppression.construct_fid(components, data.time_axis())
    assert len(components) == 4
