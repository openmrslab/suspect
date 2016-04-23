from suspect.fitting import singlet
from suspect import basis, MRSData

import numpy


def test_gaussian():
    time_axis = numpy.arange(0, 0.512, 5e-4)
    fid = basis.gaussian(time_axis, 0, 0, 50.0)
    data = MRSData(fid, 5e-4, 123)
    model = singlet.Model({
        "g1": {
            "amplitude": 1.0,
            "fwhm": 50.0,
            "phase": "0",
            "frequency": 0.0
        }
    })
    fitting_result = model.fit(data)
    numpy.testing.assert_allclose(fitting_result["params"]["g1"]["fwhm"], 50.0)
    numpy.testing.assert_allclose(fitting_result["params"]["g1"]["amplitude"], 1.0)
    numpy.testing.assert_allclose(fitting_result["params"]["g1"]["frequency"], 0.0, atol=1e-7)

    numpy.testing.assert_allclose(fitting_result["fit"], fid)


# def test_multiple_gaussians():
#     time_axis = numpy.arange(0, 0.512, 5e-4)
#     g1 = basis.gaussian(time_axis, 0, 0, 50.0)
#     g2 = basis.gaussian(time_axis, 250, 0, 40.0)
#     fid = g1 + g2
#     data = MRSData(fid, 5e-4, 123)
#     model = singlet.Model({
#         "g1": {
#             "amplitude": 1.0,
#             "fwhm": 50.0,
#             "phase": "0",
#             "frequency": {"value": 0.0, "min": -50, "max": 50}
#         },
#         "g2": {
#             "amplitude": "g1amplitude",
#             "fwhm": 45,
#             "phase": "0",
#             "frequency": 230
#         }
#     })
#
#     fitting_result = model.fit(data)
#     params = fitting_result["params"]
#
#     print(params["g1"])
#     numpy.testing.assert_allclose(params["g1"]["fwhm"], 50.0)
#     numpy.testing.assert_allclose(params["g1"]["amplitude"], 1.0)
#     numpy.testing.assert_allclose(params["g1"]["frequency"], 0.0, atol=1e-7)
#     numpy.testing.assert_allclose(params["g2"]["fwhm"], 40.0)
#     numpy.testing.assert_allclose(params["g2"]["amplitude"], 1.0)
#     numpy.testing.assert_allclose(params["g2"]["frequency"], 250.0)
#
#     numpy.testing.assert_allclose(fitting_result["fit"], fid)
#     numpy.testing.assert_allclose(fitting_result["fit_components"]["g1"], g1)
#     numpy.testing.assert_allclose(fitting_result["fit_components"]["g2"], g2)
