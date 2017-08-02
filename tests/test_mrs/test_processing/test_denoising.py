import suspect

import numpy as np

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def test_sift_preserves_dtype():
    time_axis = np.arange(0, 1.024, 1e-3)
    input_signal = suspect.basis.gaussian(time_axis, 0, 0, 35)
    input_signal += np.random.randn(1024) * 0.00001
    complex_denoise = suspect.processing.denoising.sift(input_signal, 0.001)
    assert complex_denoise.dtype == np.complex128
    real_denoise = suspect.processing.denoising.sift(np.real(input_signal), 0.001)
    assert real_denoise.dtype == np.float64
    np.testing.assert_allclose(np.real(complex_denoise), real_denoise)


def test_spline():
    # we need to check if this runs correctly when number of splines is not a
    # factor of length of signal, so that padding is required.
    # generate a sample signal
    input_signal = np.random.randn(295) + 10
    # denoise the signal with splines
    output_signal = suspect.processing.denoising.spline(input_signal, 32, 2)
    # main thing is that the test runs without errors, but we can also check
    # for reduced std in the result
    assert np.std(output_signal) < np.std(input_signal)


def test_wavelet():
    # this is to check if the code runs without throwing double -> integer
    # conversion issues
    # generate a sample signal
    input_signal = np.random.randn(295) + 10
    # denoise the signal with splines
    output_signal = suspect.processing.denoising.wavelet(input_signal, "db8", 1e-2)
    # main thing is that the test runs without errors, but we can also check
    # for reduced std in the result
    assert np.std(output_signal) < np.std(input_signal)