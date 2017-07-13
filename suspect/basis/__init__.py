import numpy
from ..mrsobjects import MRSData


def gaussian(time_axis, frequency, phase, fwhm, f0=123.0):
    dt = time_axis[1] - time_axis[0]
    oscillatory_term = numpy.exp(2j * numpy.pi * (frequency * time_axis) + 1j * phase)
    damping = numpy.exp(-time_axis ** 2 / 4 * numpy.pi ** 2 / numpy.log(2) * fwhm ** 2)
    fid = oscillatory_term * damping
    fid[0] /= 2.0
    # normalise the fid so the peak has area 1
    # this works because the area of the peak (ignoring phase effects)
    # is fid[0] (=0.5) * np (FT tells us fid[0] is mean value of spectrum)
    # but we really want the frequency-amplitude product to be 1 so that
    # the chosen df does not affect the area, so we divide by df, which is
    # equivalent to multiplying by dt * np, then the np terms cancel and we
    # are left with the dt term (and a 2 because fid[0] = 0.5, not 1)
    fid = fid * dt * 2.0
    return MRSData(fid, dt, f0)


def lorentzian(time_axis, frequency, phase, fwhm, f0=123.0):
    oscillatory_term = numpy.exp(1j * (2 * numpy.pi * frequency * time_axis + phase))
    damping = numpy.exp(-time_axis * numpy.pi * fwhm)
    fid = oscillatory_term * damping
    return fid / len(time_axis)
