import suspect
import numpy as np


def test_mag_real_zero():
    time_axis = np.arange(0, 1.024, 2.5e-4)
    sample_data = (6 * suspect.basis.gaussian(time_axis, 0, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 250, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 700, 0.0, 12))
    sample_data = sample_data.adjust_phase(0.2, 0)
    sample_data += np.random.rand(len(sample_data)) * 1e-6

    phi0, phi1 = suspect.processing.phase.mag_real(sample_data)

    np.testing.assert_allclose(phi0, -0.2, rtol=0.4)


def test_acme_zero():
    time_axis = np.arange(0, 1.024, 2.5e-4)
    sample_data = (6 * suspect.basis.gaussian(time_axis, 0, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 50, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 200, 0.0, 12))
    sample_data = sample_data.adjust_phase(0.2, 0)
    sample_data += np.random.rand(len(sample_data)) * 1e-6

    phi0, phi1 = suspect.processing.phase.acme(sample_data)

    np.testing.assert_allclose(phi0, -0.2, rtol=0.05)


def test_acme_first():
    time_axis = np.arange(0, 1.024, 2.5e-4)
    sample_data = (6 * suspect.basis.gaussian(time_axis, 0, 0.0, 6)
                   + suspect.basis.gaussian(time_axis, 150, 0.0, 6))
    sample_data += np.random.rand(len(sample_data)) * 2e-6

    in_0 = 0.5
    in_1 = 0.001
    sample_data = sample_data.adjust_phase(in_0, in_1)

    out_0, out_1 = suspect.processing.phase.acme(sample_data)

    np.testing.assert_allclose(in_0, -out_0, rtol=0.05)
    np.testing.assert_allclose(in_1, -out_1, rtol=0.05)


def test_acme_range_hz():
    time_axis = np.arange(0, 1.024, 2.5e-4)
    sample_data = (6 * suspect.basis.gaussian(time_axis, 0, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 50, 0.0, 12)
                   - suspect.basis.gaussian(time_axis, 200, 0.0, 12))
    sample_data += np.random.rand(len(sample_data)) * 1e-7

    in_0 = 0.2
    in_1 = 0.001
    sample_data = sample_data.adjust_phase(in_0, in_1)

    out_0, out_1 = suspect.processing.phase.acme(sample_data, range_hz=(-1000, 75))

    np.testing.assert_allclose(in_0, -out_0, rtol=0.05)
    np.testing.assert_allclose(in_1, -out_1, rtol=0.2)


def test_real():
    mega = suspect.io.load_twix("/Users/ben/mrs/openmrslab/misc_suspect/data/JESS_MEGAPRESS/full_example_2/MRS_pre/meas_MID00155_FID16351_MEGAPRESS_68_Vis.dat")
    tmp_ave = np.mean(mega, axis=(0, 1))
    channel_weights = suspect.processing.channel_combination.svd_weighting(tmp_ave)
    cc_mega = suspect.processing.channel_combination.combine_channels(mega, channel_weights)
    on, off = np.mean(cc_mega, axis=0)
    import matplotlib.pyplot as plt
    plt.plot(off.spectrum())
    print(suspect.processing.phase.mag_real(off))
    print(suspect.processing.phase.acme(off))
    plt.plot(off.adjust_phase(*suspect.processing.phase.acme(off, range_ppm=(4.2, 0))).spectrum())
    plt.plot(off.adjust_phase(*suspect.processing.phase.mag_real(off, range_ppm=(4.2, 0))).spectrum())
    plt.show()
    #assert 3==4
