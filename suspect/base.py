import numpy


class ImageBase(numpy.ndarray):
    """
    numpy.ndarray subclass with an affine transform associated with it
    """
    def __new__(cls, input_array, transform=None):
        # input_array is an already formed ndarray
        # we want to make it our class type
        obj = numpy.asarray(input_array).view(cls)
        obj.transform = transform
        return obj

    def __array_finalize__(self, obj):
        self.transform = getattr(obj, 'transform', None)

    def to_scanner(self, x, y, z):
        """
        Converts a 3d position in ImageBase space to the scanner
        reference space. Raises a ValueError if no transform is set.

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

        transformed_point = self.transform * numpy.matrix([x, y, z, 1]).T

        return numpy.squeeze(numpy.asarray(transformed_point))[0:3]

    def from_scanner(self, x, y, z):
        """
        Converts a 3d position in scanner space to the ImageBase
        reference space. Raises a ValueError if no transform is set.

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

        transformed_point = numpy.linalg.inv(self.transform) * numpy.matrix([x, y, z, 1]).T

        return numpy.squeeze(numpy.asarray(transformed_point))[0:3]

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
