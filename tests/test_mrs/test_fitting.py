from suspect.fitting import singlet
from suspect import basis, MRSData

import numpy


def test_gaussian():
    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 50.0) + 0.00001 * (numpy.random.rand(1024) - 0.5)
    data = MRSData(fid, 5e-4, 123)
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

    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["width"], 50.0, rtol=5e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["amplitude"], 1.0, rtol=2e-2)
    numpy.testing.assert_allclose(fitting_result["model"]["pcr"]["frequency"], 0.0, atol=1e-2)

    numpy.testing.assert_allclose(fitting_result["fit"], fid, atol=0.001)
