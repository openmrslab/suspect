import suspect.base

import numpy


class MRSBase(suspect.base.ImageBase):
    """
    numpy.ndarray subclass with additional metadata like sampling rate and echo
    time.

    """
    def __new__(cls, input_array, dt, f0, te=30, tr=-1, ppm0=4.7, voxel_dimensions=(10, 10, 10), transform=None, metadata=None):
        obj = super(MRSBase, cls).__new__(cls, input_array, transform)
        # add the new attributes to the created instance
        obj._dt = dt
        obj._f0 = f0
        obj._te = te
        obj._tr = tr
        obj.ppm0 = ppm0
        obj.voxel_dimensions = voxel_dimensions
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        # if this instance is being created by slicing from another MRSBase, copy the parameters across
        self._dt = getattr(obj, 'dt', None)
        self._f0 = getattr(obj, 'f0', None)
        self._te = getattr(obj, 'te', 30)
        self._tr = getattr(obj, 'tr', -1)
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
        cast_array = new_array.view(type(self))
        cast_array._dt = self.dt
        cast_array._f0 = self.f0
        cast_array._te = self.te
        cast_array._tr = self.tr
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
    def tr(self):
        """The repetition time of the sequence in ms.

        """
        return self._tr

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

    def voxel_volume(self):
        """

        Returns
        -------
        float
            The size of the voxel in cubic mm.

        """
        return numpy.prod(self.voxel_size)

    def slice_hz(self, lower_bound, upper_bound):
        """
        Creates a slice object to access the region of the spectrum between
        the specified bounds, in Hertz.

        Parameters
        ----------
        lower_bound : float
            The lower frequency bound of the region, in Hertz.
        upper_bound : float
            The upper frequency bound of the region, in Hertz.

        Returns
        -------
        out : Slice
        """
        lower_index = numpy.floor((lower_bound + self.sw / 2) / self.df)
        upper_index = numpy.ceil((upper_bound + self.sw / 2) / self.df)
        if lower_index < 0:
            raise ValueError("Could not create a slice for lower bound {}, value is outside range".format(lower_bound))
        if upper_index < 0:
            raise ValueError("Could not create a slice for upper bound {}, value is outside range".format(upper_bound))
        return slice(int(lower_index), int(upper_index))

    def slice_ppm(self, lower_bound, upper_bound):
        """
        Creates a slice object to access the region of the spectrum between
        the specified bounds, in PPM.

        Parameters
        ----------
        lower_bound : float
            The lower frequency bound of the region, in PPM.
        upper_bound : float
            The upper frequency bound of the region, in PPM.

        Returns
        -------
        out : Slice
        """
        return self.slice_hz(self.ppm_to_hertz(lower_bound),
                             self.ppm_to_hertz(upper_bound))

    @property
    @suspect.base.requires_transform
    def centre(self):
        # unlike the ImageBase class, the centre of spectroscopy volumes is
        # already encoded in the position of the volume
        return self.position


class MRSData(MRSBase):
    """
    MRS data in the time domain.
    """

    def fid(self):
        """
        Returns itself. This is useful when you have either a spectrum or an
        FID, but want an FID
        
        Returns
        -------
        MRSData:
            The called MRSData object
        """
        return self

    def spectrum(self):
        """
        Returns
        -------
        MRSSpectrum
            The Fourier-transformed and shifted data, represented as a spectrum

        """
        spectrum = self.inherit(numpy.fft.fftshift(numpy.fft.fft(self, axis=-1), axes=-1)).view(MRSSpectrum)
        return spectrum

    def adjust_phase(self, zero_phase, first_phase=0., fixed_frequency=0.):
        """
        Adjust the phases of the signal.

        Refer to suspect.adjust_phase for full documentation.

        Parameters
        ----------
        zero_phase: float
            The zero order phase shift in radians
        first_phase: float
            The first order phase shift in radians per Hertz
        fixed_frequency: float
            The frequency at which the first order phase shift is zero

        Returns
        -------
        out : MRSData
            Phase adjusted FID

        See Also
        --------
        suspect.adjust_phase : equivalent function
        """
        # easiest to do this in the spectral domain
        spectrum = self.spectrum()
        return spectrum.adjust_phase(zero_phase, first_phase, fixed_frequency).fid()

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


class MRSSpectrum(MRSBase):
    """
    MRS data in the frequency domain
    """

    def spectrum(self):
        """
        Returns itself. This is useful if you have something which is either a
        spectrum or an FID, but want a spectrum.
        Returns
        -------
        MRSSpectrum
            The called MRSpectrum object.
        """
        return self

    def fid(self):
        """
        Returns
        -------
        MRSData
            The inverse-Fourier-shifted and inverse-Fourier-transformed data, represented as a FID
        """
        fid = self.inherit(numpy.fft.ifft(numpy.fft.ifftshift(self, axes=-1), axis=-1)).view(MRSData)
        return fid

    def adjust_phase(self, zero_phase, first_phase=0., fixed_frequency=0.):
        """
        Adjust the phases of the signal.

        Refer to suspect.adjust_phase for full documentation.

        Parameters
        ----------
        zero_phase: float
            The zero order phase shift in radians
        first_phase: float
            The first order phase shift in radians per Hertz
        fixed_frequency: float
            The frequency at which the first order phase shift is zero

        Returns
        -------
        out : MRSSpectrum
            Phase adjusted spectrum

        See Also
        --------
        suspect.adjust_phase : equivalent function
        """
        phase_ramp = numpy.linspace(-self.sw / 2,
                                    self.sw / 2,
                                    self.np,
                                    endpoint=False)
        phase_shift = zero_phase + first_phase * (fixed_frequency + phase_ramp)
        phased_spectrum = self * numpy.exp(1j * phase_shift)
        return phased_spectrum

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
        out : MRSSpectrum
            Frequency adjusted spectrum

        See Also
        --------
        suspect.adjust_frequency : equivalent function
        """
        return self.fid().adjust_frequency(frequency_shift).spectrum()
