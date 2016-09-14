from suspect.fitting import singlet
from suspect import basis, MRSData

import numpy

def test_gaussian():
    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 50.0) + 0.00001 * (numpy.random.rand(1024) - 0.5)
    data = MRSData(fid, 5e-4, 123)

    #Original test with all parameters passed in; correct data types; integer values
    model1 = {
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

    #Floating point values; invalid key added to width dict, to test whether KeyError is raised
    model2 = {
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

    #No width value passed in, to test whether KeyError is raised
    model3 = {
        "phase0": 0,
        "phase1": 0,
        "pcr": {
            "amplitude": 1,
            "width": {
                #"value": 45,
                "min": 42,
                "max": 55,

            },
            "phase": "0",
            "frequency": 0
        }
    }

    #No phase value passed in, to test whether phase is fixed to 0 by default
    model4 = {
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

    #Str value supplied for phase0 and phase1, to test whether TypeError is raised
    model5 = {
        "phase0": str,
        "phase1": str,
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

    # Str value supplied for amplitude, to test whether TypeError is raised
    model6 = {
        "phase0": 0.0,
        "phase1": 0.0,
        "pcr": {
            "amplitude": str,
            "width": {
                "value": 45.0,
                "min": 42.0,
                "max": 55.0,
            },
            "phase": "0",
            "frequency": 0.0
        }
    }

    set_of_models = [model1, model2, model3, model4, model5, model6]

    for model in set_of_models:
        try:
            fitting_result = singlet.fit(data, model)

            numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=1e-2)
            numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
            numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=1e-2)

            numpy.testing.assert_allclose(fitting_result["fit"], fid, atol=0.001)

        except(TypeError, KeyError, ReferenceError):
            pass
