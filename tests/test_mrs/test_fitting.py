from suspect.fitting import singlet
from suspect import basis, MRSData

import numpy as np
import pytest
import random

np.random.seed(1024)


@pytest.fixture
def single_peak():
    time_axis = np.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 10.0) + 0.00001 * (np.random.rand(1024) - 0.5)
    return fid


@pytest.fixture
def correlated_peaks(single_peak):
    fid2 = 0.67 * basis.gaussian(single_peak.time_axis(), 200, 0, 15.0)
    return single_peak + fid2


def test_from_dict_single_peak(single_peak):
    model_dict = {
        "phase0": 0,
        "phase1": "0",
        "pcr": {
            "amplitude": 0.5,
            "frequency": 0,
            "phase": "0"
        }
    }

    model = singlet.Model.from_dict(model_dict)
    print(model.composite_model.param_hints)
    assert model.composite_model.param_hints["pcr_phase"] == {"value": 0.0, "vary": False}

    result = model.fit(single_peak)

    np.testing.assert_almost_equal(result.params["pcr_amplitude"], 1, 3)
    np.testing.assert_almost_equal(result.params["pcr_fwhm"], 10, 2)


def test_from_dict_double_peak(correlated_peaks):
    model_dict = {
        "cr1": {
        },
        "cr2": {
            "frequency": 200,
            "amplitude": {
                "expr": "cr1_amplitude*0.67"
            }
        }
    }

    model = singlet.Model.from_dict(model_dict)

    result = model.fit(correlated_peaks)

    np.testing.assert_almost_equal(result.params["cr1_amplitude"], 1, 3)
    np.testing.assert_almost_equal(result.params["cr2_amplitude"], 0.67, 3)

