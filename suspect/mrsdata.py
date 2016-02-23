import numpy


class MRSData(numpy.ndarray):

    def __new__(cls, input_array, dt, f0, te=30, ppm0=4.7, voxel_dimensions=(10, 10, 10)):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj.dt = dt
        obj.f0 = f0
        obj.te = te
        obj.ppm0 = ppm0
        obj.np = obj.shape[-1]
        obj.df = 1.0 / dt / obj.np
        obj.voxel_dimensions = voxel_dimensions
        return obj

    def __array_finalize__(self, obj):
        # if this instance is being created by slicing from another MRSData, copy the parameters across
        self.dt = getattr(obj, 'dt', None)
        self.f0 = getattr(obj, 'f0', None)
        self.te = getattr(obj, 'te', 30)
        self.ppm0 = getattr(obj, 'ppm0', None)
        self.df = getattr(obj, 'df', None)
        self.np = getattr(obj, 'np', None)
        #self.np = self.shape[-1]
        self.voxel_dimensions = getattr(obj, 'voxel_dimensions', (10, 10, 10))

    def inherit(self, new_array):
        cast_array = new_array.view(MRSData)
        cast_array.dt = self.dt
        cast_array.f0 = self.f0
        cast_array.np = self.np
        cast_array.df = self.df
        cast_array.te = self.te
        cast_array.ppm0 = self.ppm0
        cast_array.voxel_dimensions = self.voxel_dimensions
        return cast_array

    def time_axis(self):
        return numpy.arange(0.0, self.dt * self.np, self.dt)

    def frequency_axis(self):
        sw = 1.0 / self.dt
        return numpy.linspace(-sw / 2, sw / 2, self.np, endpoint=False)

    def voxel_size(self):
        return numpy.prod(self.voxel_dimensions)
