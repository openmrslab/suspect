import pyflo.component
import suspect.mrs.processing
import suspect.mrs

from collections import deque
import numpy
import numbers


class BeginGroup:
    pass


class EndGroup:
    pass


class SVDChannelWeights(pyflo.component.Component):

    def __init__(self, metadata):
        inports = {
            "in": {},
        }
        outports = {
            "weights": {}
        }
        super().__init__(inports, outports)
        self.inports["in"].on('data', self.calculate_channel_weights)

    def calculate_channel_weights(self, data):
        weights = suspect.mrs.processing.svd_weighting(data)
        self.outports["weights"].send_data(weights)


class WeightedAverage(pyflo.component.Component):
    axis = None

    def __init__(self, metadata):
        inports = {
            "in": {},
            "weights": {},
            "axis": {},
        }
        outports = {
            "out": {}
        }
        super().__init__(inports, outports)
        self.inports["in"].on('data', self.data)
        self.queued_data = deque()
        self.inports["weights"].on('data', self.weights)
        self.queued_weights = deque()
        self.inports["axis"].on('data', self.set_axis)

    def data(self, data):
        self.queued_data.append(data)
        self.combine()

    def weights(self, data):
        self.queued_weights.append(data)
        self.combine()

    def combine(self):
        # options for combining data
        # 1: axis not attached, weights not attached, data in queue
        # 2: axis set, weights not attached, data in queue
        # 3: axis not attached, weights in queue, data in queue
        # 4: axis set, weights in queue, data in queue
        if self.inports["axis"].attached():
            if self.axis is None:
                return
            else:
                axis = int(self.axis)
        else:
            axis = 0
        if self.inports["weights"].attached():
            while len(self.queued_data) > 0 and len(self.queued_weights) > 0:
                data_to_combine = self.queued_data.popleft()
                weights = self.queued_weights.popleft()
                combined_data = data_to_combine.inherit(numpy.average(data_to_combine, axis=axis, weights=weights))
                self.outports["out"].send_data(combined_data)
        else:
            while len(self.queued_data) > 0:
                data_to_combine = self.queued_data.popleft()
                combined_data = data_to_combine.inherit(numpy.average(data_to_combine, axis=axis))
                self.outports["out"].send_data(combined_data)

    def set_axis(self, axis):
        self.axis = axis
        self.combine()


class WaterPeakAlignment(pyflo.component.Component):
    def __init__(self, metadata):
        inports = {
            "in": {}
        }
        outports = {
            "shift": {}
        }
        super().__init__(inports, outports)
        self.inports["in"].on("beginGroup", self.outports["shift"].begin_group)
        self.inports["in"].on("data", self.data)
        self.inports["in"].on("endGroup", self.outports["shift"].end_group)

    def data(self, data):
        frequency_shift = suspect.mrs.processing.frequency_correction.residual_water_alignment(data)
        self.outports["shift"].send_data(frequency_shift)


class FrequencyShift(pyflo.component.Component):
    def __init__(self, metadata):
        inports = {
            "in": {},
            "shift": {},
        }
        outports = {
            "out": {},
        }
        super().__init__(inports, outports)
        self.inports["in"].on("beginGroup", self.in_begin_group)
        self.inports["in"].on("data", self.data)
        self.inports["in"].on("endGroup", self.in_end_group)
        self.in_queue = deque()
        self.inports["shift"].on("beginGroup", self.shift_begin_group)
        self.inports["shift"].on("data", self.shift)
        self.inports["shift"].on("endGroup", self.shift_end_group)
        self.shift_queue = deque()

    def in_begin_group(self):
        self.in_queue.append(BeginGroup())

    def data(self, data):
        self.in_queue.append(data)
        self.process()

    def in_end_group(self):
        self.in_queue.append(EndGroup())
        self.process()

    def shift_begin_group(self):
        self.shift_queue.append(BeginGroup())

    def shift(self, shift):
        self.shift_queue.append(shift)
        self.process()

    def shift_end_group(self):
        self.shift_queue.append(EndGroup())
        self.process()

    def process(self):
        while len(self.in_queue) and len(self.shift_queue):
            in_val = self.in_queue.popleft()
            shift_val = self.shift_queue.popleft()
            if isinstance(in_val, BeginGroup) and isinstance(shift_val, BeginGroup):
                self.outports["out"].begin_group()
            elif isinstance(in_val, EndGroup) and isinstance(shift_val, EndGroup):
                self.outports["out"].end_group()
            elif type(in_val) == suspect.mrs.MRSData and isinstance(shift_val, numbers.Number):
                print("correcting frequency shift {}".format(shift_val))
                transformed_fid = suspect.mrs.processing.frequency_correction.transform_fid(in_val, shift_val, 0)
                self.outports["out"].send_data(transformed_fid)
            else:
                raise Exception("in {} and shift {} values didn't match".format(in_val, shift_val))


import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot


class TempPlot(pyflo.component.Component):
    def __init__(self, metadata):
        inports = {
            "in": {},
        }
        super().__init__(inports, {})
        self.inports["in"].on('data', self.data)
        self.inports["in"].on('beginGroup', self.begin)
        self.inports["in"].on('endGroup', self.end)

    def begin(self):
        pass

    def data(self, data):
        spectrum = numpy.fft.fftshift(numpy.fft.fft(data))
        pyplot.plot(numpy.abs(spectrum))
        pyplot.savefig("/home/ben/sample_plot.pdf")

    def end(self):
        print("saving figure")
        pyplot.savefig("/home/ben/sample_plot.pdf")
