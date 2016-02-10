import suspect.pyflo.processing as p
import pyflo.ports
import suspect.mrs

import pytest
import numpy
from unittest.mock import Mock


@pytest.fixture
def simple_data():
    source_array = numpy.ones((4, 128), 'complex')
    source_array[1, :] *= 2
    source_array[2, :] *= 4
    source_array[3, :] *= 8
    data = suspect.mrs.MRSData(source_array, 5e-4, 123)
    return data


@pytest.fixture
def complex_data():
    source_array = numpy.ones((8, 4, 128), 'complex')
    data = suspect.mrs.MRSData(source_array, 5e-4, 123)
    return data


def test_channel_weights_no_axis(simple_data):
    component = p.SVDChannelWeights({})
    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])
    #target_port = Mock()
    #print(component.outports)
    #component.outports["out"].connect(target_port)
    data_port.send_data(simple_data)
    #print(target_port.call_args)


def test_channel_weights_data(simple_data):
    component = p.SVDChannelWeights({})
    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])

    target_port = pyflo.ports.Inport({"name": "result"})
    mock = Mock()
    target_port.on('data', mock)
    component.outports["weights"].connect(target_port)

    data_port.send_data(simple_data)
    result = mock.call_args[0][0]
    assert result.shape == (4,)
    numpy.testing.assert_almost_equal(result[0] / result[1], simple_data[0, 0] / simple_data[1, 0])


def test_average_data_only(simple_data):
    component = p.WeightedAverage({})
    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])
    target_port = pyflo.ports.Inport({"name": "result"})
    mock = Mock()
    target_port.on('data', mock)
    component.outports["out"].connect(target_port)
    data_port.send_data(simple_data)
    result = mock.call_args[0][0]
    assert result.shape == (128,)
    assert result[0] == 3.75
    assert result.dt == 5e-4
    assert result.f0 == 123


def test_average_data_weights(simple_data):
    component = p.WeightedAverage({})
    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])
    weights_port = pyflo.ports.Outport({"name": "weights"})
    weights_port.connect(component.inports["weights"])
    target_port = pyflo.ports.Inport({"name": "result"})
    mock = Mock()
    target_port.on("data", mock)
    component.outports["out"].connect(target_port)
    data_port.send_data(simple_data)

    # component should wait for weights to be send
    mock.assert_not_called()

    weights_port.send_data(numpy.array([0, 0, 0, 1]))

    result = mock.call_args[0][0]
    assert result.shape == (128,)
    assert result[0] == 8
    assert result.dt == 5e-4
    assert result.f0 == 123


def test_average_data_axis(complex_data):
    component = p.WeightedAverage({})
    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])
    axis_port = pyflo.ports.Outport({"name": "axis"})
    axis_port.connect(component.inports["axis"])
    target_port = pyflo.ports.Inport({"name": "result"})
    mock = Mock()
    target_port.on("data", mock)
    component.outports["out"].connect(target_port)

    data_port.send_data(complex_data)

    # component should wait for weights to be send
    mock.assert_not_called()

    axis_port.send_data(1)

    result = mock.call_args[0][0]
    assert result.shape == (8, 128)
    assert result[0, 0] == 1
    assert result.dt == 5e-4
    assert result.f0 == 123


def test_residual_water_alignment():
    component = p.WaterPeakAlignment(None)

    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])

    test_spectrum = numpy.zeros(128, 'complex')
    test_spectrum[16] = 1
    test_fid = numpy.fft.ifft(test_spectrum)
    test_data = suspect.mrs.MRSData(test_fid, 1.0 / 128, 123)

    target_port = pyflo.ports.Inport({"name": "target"})
    component.outports["shift"].connect(target_port)
    mock = Mock()
    target_port.on('data', mock)

    data_port.send_data(test_data)

    mock.assert_called_once_with(16)


def test_frequency_shift():
    component = p.FrequencyShift(None)

    data_port = pyflo.ports.Outport({"name": "data"})
    data_port.connect(component.inports["in"])
    shift_port = pyflo.ports.Outport({"name": "shift"})
    shift_port.connect(component.inports["shift"])

    test_spectrum = numpy.zeros(128, 'complex')
    test_spectrum[0] = 1
    target_fid = numpy.fft.ifft(test_spectrum)
    target_data = suspect.mrs.MRSData(target_fid, 1.0 / 128, 123)
    shifted_spectrum = numpy.roll(test_spectrum, 16)
    shifted_fid = numpy.fft.ifft(shifted_spectrum)
    shifted_data = suspect.mrs.MRSData(shifted_fid, 1.0 / 128, 123)

    target_port = pyflo.ports.Inport({"name": "result"})
    component.outports["out"].connect(target_port)
    mock = Mock()
    target_port.on('data', mock)

    data_port.send_data(shifted_data)
    mock.assert_not_called()
    shift_port.send_data(-16.0)
    numpy.testing.assert_almost_equal(target_data, mock.call_args[0][0])

    # try sending the data the other way
    mock = Mock()
    target_port.on('data', mock)
    shift_port.send_data(-16.0)
    mock.assert_not_called()
    data_port.send_data(shifted_data)
    numpy.testing.assert_almost_equal(target_data, mock.call_args[0][0])
