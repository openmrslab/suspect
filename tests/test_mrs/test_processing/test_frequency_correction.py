import suspect

import numpy as np


def test_windowed_spectral_registration():
    time_axis = np.arange(0, 0.512, 5e-4)

    # construct a pair of fids with a large aligned peak at 100Hz and a smaller
    # peak offset by 5 * df. If the big peak is not excluded that will override
    # the offset peak
    target_fid = suspect.MRSData(suspect.basis.gaussian(time_axis, 0, 0, 10.0) +
                                 suspect.basis.gaussian(time_axis, 100, 0, 10.0) * 10,
                                 5e-4, 123)

    moving_fid = suspect.MRSData(suspect.basis.gaussian(time_axis, 5 * target_fid.df, 0, 10.0) +
                                 suspect.basis.gaussian(time_axis, 100, 0, 10.0) * 10,
                                 5e-4, 123)

    # ensure that the un-windowed version gives the correct answer (of 0)
    fs, ps = suspect.processing.frequency_correction.spectral_registration(moving_fid,
                                                                           target_fid)
    np.testing.assert_allclose(fs, 0, atol=0.3)

    # first we test supplying a Hz frequency range
    fs, ps = suspect.processing.frequency_correction.spectral_registration(moving_fid,
                                                                           target_fid,
                                                                           frequency_range=(-50, 50))
    np.testing.assert_allclose(fs, 5 * target_fid.df, atol=0.3)

    # next we test using a ppm slice
    fs, ps = suspect.processing.frequency_correction.spectral_registration(moving_fid,
                                                                           target_fid,
                                                                           frequency_range=target_fid.slice_ppm(5, 4.4))
    np.testing.assert_allclose(fs, 5 * target_fid.df, atol=0.3)

    # finally we test using a mask in the frequency domain
    spectral_mask = np.zeros_like(target_fid)
    spectral_mask[490:534] = 1

    fs, ps = suspect.processing.frequency_correction.spectral_registration(moving_fid,
                                                                           target_fid,
                                                                           frequency_range=spectral_mask)

    np.testing.assert_allclose(fs, 5 * target_fid.df, atol=0.3)
