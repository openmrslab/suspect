import numpy


def transform_fid(fid, frequency_shift, phase_shift):
    time_axis = fid.time_axis()
    correction = numpy.exp(2j * numpy.pi * (frequency_shift * time_axis + phase_shift))
    transformed_fid = numpy.multiply(fid, correction)
    return transformed_fid


def residual_water_alignment(data):
    current_spectrum = numpy.fft.fft(data)
    peak_index = numpy.argmax(numpy.abs(current_spectrum))
    if peak_index > len(data) / 2:
        peak_index -= len(data)
    return peak_index * data.df
