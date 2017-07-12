import numpy
import functools

from . import _transforms


def requires_transform(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.transform is None:
            raise ValueError("No transform set for {} object {}".format(type(self), self))
        return func(self, *args, **kwargs)
    return wrapper


class ImageBase(numpy.ndarray):
    """
    numpy.ndarray subclass with an affine transform associated with it
    """
    def __new__(cls, input_array, transform=None):
        # input_array is an already formed ndarray
        # we want to make it our class type
        obj = numpy.asarray(input_array).view(cls)
        if transform is not None:
            obj.transform = transform.copy()
        return obj

    def __array_finalize__(self, obj):
        self.transform = getattr(obj, 'transform', None)

    @requires_transform
    def to_scanner(self, *args):
        """
        Converts a 3d position in ImageBase space to the scanner
        reference space. Argument can either be 3 individual floats
        for x, y and z or a numpy array_like with final dimension of
        size 3.
        
        Raises a ValueError if no transform is set.

        Parameters
        ----------
        x : float
            The x coordinate of the point
        y : float
            The y coordinate of the point
        z : float
            The z coordinate of the point

        Returns
        -------
        numpy.ndarray
            The transformed 3d point in scanner coordinates
        """
        positions = _transforms.normalise_positions_for_transform(*args)

        transformed_point = numpy.einsum("ij,...j", self.transform, positions)

        return numpy.squeeze(numpy.asarray(transformed_point))[..., 0:3]

    @requires_transform
    def from_scanner(self, *args):
        """
        Converts a 3d position in scanner space to the ImageBase
        reference space. Argument can either be 3 individual floats
        for x, y and z or a numpy array_like with final dimension of
        size 3.
        
        Raises a ValueError if no transform is set.

        Parameters
        ----------
        x : float
            The x coordinate of the point
        y : float
            The y coordinate of the point
        z : float
            The z coordinate of the point

        Returns
        -------
        numpy.ndarray
            The transformed 3d point in ImageBase coordinates
        """
        positions = _transforms.normalise_positions_for_transform(*args)

        transformed_point = numpy.einsum("ij,...j",
                                         numpy.linalg.inv(self.transform),
                                         positions)

        return numpy.squeeze(numpy.asarray(transformed_point))[..., 0:3]

    @property
    @requires_transform
    def voxel_size(self):
        """
        The dimensions of a voxel.
        
        Returns
        -------
        numpy.ndarray
            The dimensions of a voxel along each axis.
        """
        return numpy.linalg.norm(self.transform, axis=0)[0:3]

    @property
    @requires_transform
    def position(self):
        """
        The centre of the ImageBase in scanner coordinates.
        """
        return self.transform[:3, 3]

    @property
    @requires_transform
    def slice_vector(self):
        return self.transform[:3, 2] / numpy.linalg.norm(self.transform[:3, 2])

    @property
    @requires_transform
    def row_vector(self):
        return self.transform[:3, 1] / numpy.linalg.norm(self.transform[:3, 1])

    @property
    @requires_transform
    def col_vector(self):
        return self.transform[:3, 0] / numpy.linalg.norm(self.transform[:3, 0])

    @requires_transform
    def _closest_axis(self, target_axis):
        overlap = numpy.abs(numpy.dot(target_axis, self.transform[:3, :3]))
        return self.transform[:3, numpy.argmax(overlap)]

    @property
    @requires_transform
    def axial_vector(self):
        """
        Returns the image axis which is most closely aligned with the axial
        direction. The returned vector is guaranteed to point in the positive
        axial direction, even if the original volume vector is in the
        opposite direction.
        
        Returns
        -------
        numpy.ndarray
            The most axial image axis
        """
        # dot the three candidate vectors with (0, 0, 1)
        best_axis = self._closest_axis((0, 0, 1))
        norm_axis = best_axis / numpy.linalg.norm(best_axis)
        # work out if we need to reverse the direction
        return norm_axis if norm_axis[2] > 0 else -1 * norm_axis

    @property
    @requires_transform
    def coronal_vector(self):
        """
        Returns the image axis which is most closely aligned with the coronal
        direction. The returned vector is guaranteed to point in the positive
        coronal direction, even if the original volume vector is in the
        opposite direction.

        Returns
        -------
        numpy.ndarray
            The most coronal image axis
        """
        # dot the three candidate vectors with (0, 1, 0)
        best_axis = self._closest_axis((0, 1, 0))
        norm_axis = best_axis / numpy.linalg.norm(best_axis)
        return norm_axis if norm_axis[1] > 0 else -1 * norm_axis

    @property
    @requires_transform
    def sagittal_vector(self):
        """
        Returns the image axis which is most closely aligned with the sagittal
        direction. The returned vector is guaranteed to point in the positive
        sagittal direction, even if the original volume vector is in the
        opposite direction.

        Returns
        -------
        numpy.ndarray
            The most sagittal image axis
        """
        # dot the three candidate vectors with (1, 0, 0)
        best_axis = self._closest_axis((1, 0, 0))
        norm_axis = best_axis / numpy.linalg.norm(best_axis)
        return norm_axis if norm_axis[0] > 0 else -1 * norm_axis
