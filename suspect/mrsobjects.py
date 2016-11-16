import numpy


class MRSBase(numpy.ndarray):
    """
    numpy.ndarray subclass with additional metadata like sampling rate and echo
    time.

    """
    def __new__(cls, input_array, dt, f0, te=30, ppm0=4.7, voxel_dimensions=(10, 10, 10), transform=None, metadata=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = numpy.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj._dt = dt
        obj._f0 = f0
        obj._te = te
        obj.ppm0 = ppm0
        obj.voxel_dimensions = voxel_dimensions
        obj.transform = transform
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        # if this instance is being created by slicing from another MRSBase, copy the parameters across
        self._dt = getattr(obj, 'dt', None)
        self._f0 = getattr(obj, 'f0', None)
        self._te = getattr(obj, 'te', 30)
        self.ppm0 = getattr(obj, 'ppm0', None)
        self.transform = getattr(obj, 'transform', None)
        self.metadata = getattr(obj, 'metadata', None)
        self.voxel_dimensions = getattr(obj, 'voxel_dimensions', (10, 10, 10))

    def __array_wrap__(self, obj):
        if len(obj.shape) == 0:
            return obj[()]
        else:
            return numpy.ndarray.__array_wrap__(self, obj)

    def __str__(self):
        return "<MRSBase instance f0={0}MHz TE={1}ms dt={2}ms>".format(self.f0, self.te, self.dt * 1e3)

    def inherit(self, new_array):
        """Converts a generic numpy ndarray into an MRSBase instance by copying its own MRS specific parameters.

        This is useful when performing some processing on the MRSBase object gives a bare ndarray result.

        Parameters
        ----------
        new_array : numpy ndarray
            Generic ndarray to be converted to MRSBase.

        Returns
        -------
        cast_array : MRSBase
            New MRSBase instance with data from new_array and parameters from self.

        """
        cast_array = new_array.view(MRSBase)
        cast_array._dt = self.dt
        cast_array._f0 = self.f0
        cast_array._te = self.te
        cast_array.ppm0 = self.ppm0
        cast_array.voxel_dimensions = self.voxel_dimensions
        cast_array.transform = self.transform
        cast_array.metadata = self.metadata
        return cast_array

    @property
    def dt(self):
        """The dwell time in s for the acquisition.

        """
        return self._dt

    @property
    def np(self):
        """The number of points in the FID.

        """
        return self.shape[-1]

    @property
    def sw(self):
        """The spectral width of the data in Hz. Calculated as 1 / dt.

        """
        return 1.0 / self.dt

    @property
    def df(self):
        """The frequency delta in Hz between neighbouring points in the spectrum.

        Calculated as the spectral width divided by the number of points.

        """
        return self.sw / self.np

    @property
    def te(self):
        """The echo time of the sequence in ms.

        """
        return self._te

    @property
    def f0(self):
        """The scanner frequency in MHz. Also referred to by LCModel as Hz per PPM.

        """
        return self._f0

    def hertz_to_ppm(self, frequency):
        """Converts a frequency in Hertz to the corresponding PPM for this dataset.

        Parameters
        ----------
        frequency : float
            the frequency in Hz

        Returns
        -------
        float
            The ppm value corresponding to given frequency in Hz

        """
        return self.ppm0 - frequency / self.f0

    def ppm_to_hertz(self, frequency):
        """Converts a frequency in PPM to the corresponding Hertz for this dataset.

        Parameters
        ----------
        frequency : float
            The frequency in PPM

        Returns
        -------
        float
            The frequency in Hz corresponding to the given frequency in PPM

        """
        return (self.ppm0 - frequency) * self.f0

    def time_axis(self):
        """

        Returns
        -------
        aranged ndarray
            An array of the sample times in seconds for each point in the FID.

        """
        return numpy.arange(0.0, self.dt * self.np, self.dt)

    def frequency_axis(self):
        """

        Returns
        -------
        ndarray
            An array of frequencies in Hertz ranging from -sw/2 to sw/2.

        """
        return numpy.linspace(-self.sw / 2, self.sw / 2, self.np, endpoint=False)

    def frequency_axis_ppm(self):
        """

        Returns
        -------
        ndarray
            An array of frequencies in PPM.

        """
        return numpy.linspace(self.hertz_to_ppm(-self.sw / 2.0),
                              self.hertz_to_ppm(self.sw / 2.0),
                              self.np, endpoint=False)

    def voxel_size(self):
        """

        Returns
        -------
        float
            The size of the voxel in cubic mm.

        """
        return numpy.prod(self.voxel_dimensions)

    def to_scanner(self, x, y, z):
        """Converts a 3d position in MRSBase space to the scanner reference frame

        Parameters
        ----------
        x : float
            The x dimension value of the voxel
        y : float
            The y dimension value of the voxel
        z : float
            The z dimension value of the voxel

        Returns
        -------
        squeezed ndarray
            The transformed 3d point converted to scanner reference frame

        """
        if self.transform is None:
            raise ValueError("No transform set for MRSBase object {}".format(self))

        transformed_point = self.transform * numpy.matrix([x, y, z, 1]).T

        return numpy.squeeze(numpy.asarray(transformed_point))[0:3]

    def from_scanner(self, x, y, z):
        """Converts a 3d position in the scanner reference frame to the MRSBase space

        Parameters
        ----------
        x : float
            The x-dimension
        y : float
            The y-dimension
        z : float
            The z-dimension

        Returns
        -------
        ndarray
            Squeezed ndarray representing a point in 3d MRSBase space

        """
        if self.transform is None:
            raise ValueError("No transform set for MRSBase object {}".format(self))

        transformed_point = numpy.linalg.inv(self.transform) * numpy.matrix([x, y, z, 1]).T

        return numpy.squeeze(numpy.asarray(transformed_point))[0:3]


class MRSData(MRSBase):

    def spectrum(self):
        """
        Returns
        -------
        MRSSpectrum
            The Fourier-transformed and shifted data, represented as a spectrum

        """
        spectrum = self.inherit(numpy.fft.fftshift(numpy.fft.fft(self, axis=-1), axes=-1))
        return spectrum


class MRSSpectrum(MRSBase):

    def __new__(cls, input_array, dt, f0, te=30, ppm0=4.7, voxel_dimensions=(10, 10, 10), transform=None, metadata=None):
        obj = numpy.asarray(input_array).view(cls)
        # add the new attributes to the created instance
        obj._dt = dt
        obj._f0 = f0
        obj._te = te
        obj.ppm0 = ppm0
        obj.voxel_dimensions = voxel_dimensions
        obj.transform = transform
        obj.metadata = metadata
        obj = obj.inherit(numpy.fft.fftshift(numpy.fft.fft(cls, axis=-1), axes=-1))
        return obj

    def fid(self):
        """
        Returns
        -------
        MRSData
            The inverse-Fourier-shifted and inverse-Fourier-transformed data, represented as a FID
        """
        fid = self.inherit(numpy.fft.ifft(numpy.fft.ifftshift(self, axis=-1), axes=-1))
        return fid

    def adjust_phase(self, zero_phase, first_phase=0, fixed_frequency=0):
        """
        Adjust the phases of the signal.

        Refer to suspect.adjust_phase for full documentation.

        See Also
        --------
        suspect.adjust_phase : equivalent function
        """
        # easiest to apply the phase shift in the frequency domain

        phase_ramp = numpy.linspace(-self.sw / 2,
                                    self.sw / 2,
                                    self.np,
                                    endpoint=False)
        phase_shift = zero_phase + first_phase * (fixed_frequency + phase_ramp)
        phased_spectrum = self * numpy.exp(1j * phase_shift)
        return self.inherit(numpy.fft.ifft(numpy.fft.ifftshift(phased_spectrum, axes=-1), axis=-1))

    def adjust_frequency(self, frequency_shift):
        """
        Adjust the centre frequency of the signal.

        Refer to suspect.adjust_frequency for full documentation.

        Parameters
        ----------
        frequency_shift: float
            The amount to shift the frequency, in Hertz.

        Returns
        -------
        out : MRSData
            Frequency adjusted FID

        See Also
        --------
        suspect.adjust_frequency : equivalent function
        """
        correction = numpy.exp(2j * numpy.pi * (frequency_shift * self.time_axis()))
        return self.inherit(numpy.multiply(self, correction))
