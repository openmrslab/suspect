from suspect.fitting import singlet
from suspect import basis, MRSData

import numpy
import pytest
import random

numpy.random.seed(1024)

@pytest.fixture
def fixed_fid():

    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 50.0) + 0.00001 * (numpy.random.rand(1024) - 0.5)
    return fid


@pytest.fixture
def fixed_fid_sum():

    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 50.0) + 0.00001 * (numpy.random.rand(1024) - 0.5)
    fid2 = basis.gaussian(time_axis, 200, 0, 50.0)
    return fid + fid2


def test_gaussian(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    # Original test with all parameters passed in; correct data types; integer values
    model = {
        "phase0": 0,
        "phase1": 0,
        "pcr": {
            "amplitude": 1,
            "width": {
                "value": 45,
                "min": 42,
                "max": 55
            },
            "phase": "0",
            "frequency": 0
        }
    }
    fitting_result = singlet.fit(data, model)

    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

    numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)


def test_gaussian_2(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    # Floating point values; invalid key added to width dict, to test whether KeyError is raised
    model = {
        "phase0": 0.0,
        "phase1": 0.0,
        "pcr": {
            "amplitude": 1.0,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
                "avg": 47
            },
            "phase": "0",
            "frequency": 0.0
        }
    }
    with pytest.raises(KeyError):
        fitting_result = singlet.fit(data, model)

        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

        numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)


def test_gaussian_3(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    # No width value passed in, to test whether KeyError is raised
    model = {
        "phase0": 0,
        "phase1": 0,
        "pcr": {
            "amplitude": 1,
            "width": {
                # "value": 45,
                "min": 42,
                "max": 55,

            },
            "phase": "0",
            "frequency": 0
        }
    }
    with pytest.raises(KeyError):
        fitting_result = singlet.fit(data, model)

        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

        numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)

def test_gaussian_4(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    # No phase value passed in, to test whether phase is fixed to 0 by default
    model = {
        "phase0": 0,
        "phase1": 0,
        "pcr": {
            "amplitude": 1,
            "width": {
                "value": 45,
                "min": 42,
                "max": 55,
            },
            # "phase": "0",
            "frequency": 0
        }
    }

    fitting_result = singlet.fit(data, model)

    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=5e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=5e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

    numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)


def test_gaussian_5(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    # None value supplied for phase0 and phase1, to test whether TypeError is raised
    model = {
        "phase0": None,
        "phase1": None,
        "pcr": {
            "amplitude": 1.0,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": 0.0
        }
    }
    with pytest.raises(TypeError):
        fitting_result = singlet.fit(data, model)

        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

        numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)


def test_gaussian_6(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    # None value supplied for amplitude, to test whether TypeError is raised
    model = {
        "phase0": 0.0,
        "phase1": 0.0,
        "pcr": {
            "amplitude": None,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": 0.0
        }
    }

    with pytest.raises(TypeError):
        fitting_result = singlet.fit(data, model)

        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

        numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)


def test_gaussian_dependencies(fixed_fid):

    data = MRSData(fixed_fid, 5e-4, 123)

    model = {
        "phase0": 0.0,
        "phase1": 0.0,
        "pcr": {
            "amplitude": 1.0,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": "pcr2_frequency+200"
        },
        "pcr2": {
            "amplitude": 1.0,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": "pcr_frequency-200"
        }
    }

    with pytest.raises(ReferenceError):
        fitting_result = singlet.fit(data, model)

        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
        numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

        numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid, atol=0.001)


def test_gaussian_dependencies2(fixed_fid_sum):

    data = MRSData(fixed_fid_sum, 5e-4, 123)

    model = {
        "phase0": 0.0,
        "phase1": 0.0,
        "pcr": {
            "amplitude": 1.0,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": 0
        },
        "pcr2": {
            "amplitude": 1.0,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": "pcr_frequency+200"
        }
    }

    fitting_result = singlet.fit(data, model)

    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=5e-2)

    numpy.testing.assert_allclose(fitting_result["fit"], fixed_fid_sum, atol=0.001)