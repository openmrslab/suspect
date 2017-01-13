import suspect

import numpy

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

def test_spline():
    # we need to check if this runs correctly when number of splines is not a
    # factor of length of signal, so that padding is required.
    # generate a sample signal
    input_signal = numpy.random.randn(295) + 10
    # denoise the signal with splines
    output_signal = suspect.processing.denoising.spline(input_signal, 32, 2)
    # main thing is that the test runs without errors, but we can also check
    # for reduced std in the result
    assert numpy.std(output_signal) < numpy.std(input_signal)


def test_wavelet():
    # this is to check if the code runs without throwing double -> integer
    # conversion issues
    # generate a sample signal
    input_signal = numpy.random.randn(295) + 10
    # denoise the signal with splines
    output_signal = suspect.processing.denoising.wavelet(input_signal, "db8", 1e-2)
    # main thing is that the test runs without errors, but we can also check
    # for reduced std in the result
    assert numpy.std(output_signal) < numpy.std(input_signal)