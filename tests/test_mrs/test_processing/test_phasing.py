import suspect
import numpy as np


def test_mag_real_zero():
    time_axis = np.arange(0, 1.024, 2.5e-4)
    sample_data = (6 * suspect.basis.gaussian(time_axis, 0, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 250, 0.0, 12)
                   + suspect.basis.gaussian(time_axis, 500, 0.0, 12))
    sample_data = sample_data.adjust_phase(0.2, 0)
    sample_data += np.random.rand(len(sample_data)) * 1e-6

    phi0, phi1 = suspect.processing.phase.mag_real(sample_data)

    np.testing.assert_allclose(phi0, -0.2, rtol=0.05)


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
    np.testing.assert_allclose(in_1, -out_1, rtol=0.05)
