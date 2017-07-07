import numpy

from . import _transforms


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
        if self.transform is None:
            raise ValueError("No transform set for {} object {}".format(type(self), self))

        positions = _transforms.normalise_positions_for_transform(*args)

        transformed_point = numpy.einsum("ij,...j", self.transform, positions)

        return numpy.squeeze(numpy.asarray(transformed_point))[..., 0:3]

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
        if self.transform is None:
            raise ValueError("No transform set for {} object {}".format(type(self), self))

        positions = _transforms.normalise_positions_for_transform(*args)

        transformed_point = numpy.einsum("ij,...j",
                                         numpy.linalg.inv(self.transform),
                                         positions)

        return numpy.squeeze(numpy.asarray(transformed_point))[..., 0:3]

    @property
    def voxel_size(self):
        """
        The dimensions of a voxel.
        :return:
        """
        if self.transform is None:
            raise ValueError("No transform set for {} object {}".format(type(self), self))

        return numpy.linalg.norm(self.transform, axis=0)[0:3]

    @property
    def position(self):
        """
        The centre of the ImageBase in scanner coordinates.
        """
        if self.transform is None:
            raise ValueError("No transform set for {} object {}".format(type(self), self))

        return self.transform[:3, 3]
