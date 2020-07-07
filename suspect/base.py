import numpy as np
import functools
import scipy.interpolate

from . import _transforms


def requires_transform(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.transform is None:
            raise ValueError("No transform set for {} object {}".format(type(self), self))
        return func(self, *args, **kwargs)

    return wrapper


class ImageBase(np.ndarray):
    """
    numpy.ndarray subclass with an affine transform associated with it
    """

    def __new__(cls, input_array, transform=None):
        # input_array is an already formed ndarray
        # we want to make it our class type
        obj = np.asarray(input_array).view(cls)
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

        transformed_point = np.einsum("ij,...j", self.transform, positions)

        return np.squeeze(np.asarray(transformed_point))[..., 0:3]

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

        transformed_point = np.einsum("ij,...j",
                                      np.linalg.inv(self.transform),
                                      positions)

        return np.squeeze(np.asarray(transformed_point))[..., 0:3]

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
        return np.linalg.norm(self.transform, axis=0)[0:3]

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
        return self.transform[:3, 2] / np.linalg.norm(self.transform[:3, 2])

    @property
    @requires_transform
    def row_vector(self):
        return self.transform[:3, 0] / np.linalg.norm(self.transform[:3, 0])

    @property
    @requires_transform
    def col_vector(self):
        return self.transform[:3, 1] / np.linalg.norm(self.transform[:3, 1])

    @requires_transform
    def _closest_axis(self, target_axis):
        voxel_axes = self.transform[:3, :3] / self.voxel_size
        overlap = np.abs(np.dot(target_axis, voxel_axes))
        return self.transform[:3, np.argmax(overlap)]

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
        norm_axis = best_axis / np.linalg.norm(best_axis)
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
        norm_axis = best_axis / np.linalg.norm(best_axis)
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
        norm_axis = best_axis / np.linalg.norm(best_axis)
        return norm_axis if norm_axis[0] > 0 else -1 * norm_axis

    @property
    @requires_transform
    def centre(self):
        """
        Returns the centre of the image volume in scanner coordinates.

        Returns
        -------
        numpy.ndarray
            The centre of the image volume
        :return:
        """
        return self.to_scanner((np.array(self.shape[::-1]) - 1) / 2)

    @requires_transform
    def resample(self,
                 row_vector,
                 col_vector,
                 shape,
                 centre=(0, 0, 0),
                 voxel_size=(1, 1, 1),
                 method='linear'):
        """
        Create a new volume by resampling this one using a different coordinate
        system.

        Parameters
        ----------
        row_vector: array
            Row direction vector for new volume
        col_vector: array
            Column direction vector for new volume
        shape: array
            The shape of the new volume, as slices, rows, columns
        centre: array
            The position of the centre of the new volume in scanner
            coordinates, in mm
        voxel_size: array
            The size of each voxel in the new volume, in mm
        method: str
            The interpolation method to use - either "linear" or "nearest"

        Returns
        -------
        suspect.base.ImageBase
            The resampled volume
        """
        # make sure row_vector and col_vector are normalised
        row_vector = np.asanyarray(row_vector) / np.linalg.norm(row_vector)
        col_vector = np.asanyarray(col_vector) / np.linalg.norm(col_vector)

        # mgrid produces 3D index grids for the x, y and z coords separately
        II, JJ, KK = np.mgrid[0:shape[2],
                              0:shape[1],
                              0:shape[0]].astype(np.float)
        # shift the indices from the corner to the centre
        II -= (shape[2] - 1) / 2
        JJ -= (shape[1] - 1) / 2
        KK -= (shape[0] - 1) / 2
        # scale the indices by the size of the voxel
        II *= voxel_size[0]
        JJ *= voxel_size[1]
        KK *= voxel_size[2]

        slice_vector = np.cross(row_vector, col_vector)

        # combine the x, y and z indices with the row, col and slice vectors
        # to get the spatial coordinates at each point in the new volume
        space_coords = II[..., np.newaxis] * row_vector \
                       + JJ[..., np.newaxis] * col_vector \
                       + KK[..., np.newaxis] * slice_vector + centre

        image_coords = self.from_scanner(space_coords).reshape(*space_coords.shape)[..., ::-1].astype(np.int)
        resampled = scipy.interpolate.interpn([np.arange(dim) for dim in self.shape],
                                              self,
                                              image_coords,
                                              method=method,
                                              bounds_error=False,
                                              fill_value=0).squeeze()

        transform = _transforms.transformation_matrix(row_vector,
                                                      col_vector,
                                                      space_coords[0, 0, 0],
                                                      voxel_size)

        # we have to transpose the result to go from x, y, z to row, col, slice
        return ImageBase(resampled.T, transform=transform)
