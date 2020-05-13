import numpy


def transformation_matrix(x_vector, y_vector, translation, spacing):
    """
    Creates a transformation matrix which will convert from a specified
    coordinate system to the scanner frame of reference.

    Parameters
    ----------
    x_vector : array
        The unit vector along the space X axis in scanner coordinates
    y_vector : array
        The unit vector along the space Y axis in scanner coordinates
    translation : array
        The origin of the space in scanner coordinates
    spacing : float or array
        The size of a space unit in scanner units

    Returns
    -------
    matrix : array

    """
    matrix = numpy.zeros((4, 4), dtype=numpy.float)
    matrix[:3, 0] = x_vector
    matrix[:3, 1] = y_vector
    z_vector = numpy.cross(x_vector, y_vector)
    matrix[:3, 2] = z_vector
    matrix[:3, 3] = numpy.array(translation)
    matrix[3, 3] = 1.0

    # make sure that we can append to spacing
    spacing = list(spacing)
    while len(spacing) < 4:
        spacing.append(1.0)
    for i in range(4):
        for j in range(4):
            matrix[i, j] *= spacing[j]
    return matrix


def rotation_matrix(angle, axis):
    """
    Creates a 3x3 matrix which rotates `angle` radians around `axis`
    
    Parameters
    ----------
    angle : float
        The angle in radians to rotate around the axis
    axis : array
        The unit vector around which to rotate
        
    Returns
    -------
    matrix : array
    """
    c = numpy.cos(angle)
    s = numpy.sin(angle)
    matrix = numpy.zeros((3, 3))
    matrix[0, 0] = c + axis[0] ** 2 * (1 - c)
    matrix[0, 1] = axis[0] * axis[1] * (1 - c) - axis[2] * s
    matrix[0, 2] = axis[0] * axis[2] * (1 - c) + axis[1] * s
    matrix[1, 0] = axis[1] * axis[0] * (1 - c) + axis[2] * s
    matrix[1, 1] = c + axis[1] ** 2 * (1 - c)
    matrix[1, 2] = axis[1] * axis[2] * (1 - c) - axis[0] * s
    matrix[2, 0] = axis[2] * axis[0] * (1 - c) - axis[1] * s
    matrix[2, 1] = axis[2] * axis[1] * (1 - c) + axis[0] * s
    matrix[2, 2] = c + axis[2] ** 2 * (1 - c)
    return matrix


def normalise_positions_for_transform(*args):
    """
    Takes an input set of arguments which should represent some (x, y, z)
    coords to be transformed and makes sure they are in a numpy.ndarray,
    and adds a w dimension of magnitude 1.
    
    The two acceptable forms of input arguments are a single array_like
    with final dimension of 3, or three floating point arguments representing
    x, y and z. In the first case the returned array will have the same shape
    for all axes except the last, which will go from size 3 to 4, while in the
    second case the returned array will be of shape (4,).
    
    Parameters
    ----------
    args : array_like or 3 separate floats
        The arguments to be processed
    
    Returns
    -------
    numpy.ndarray
        Points ready for transformation by a matrix
    """
    if len(args) == 3:
        positions = [*args, 1]
    elif len(args) == 1:
        positions = numpy.atleast_2d(args[0])
        w_array = numpy.expand_dims(numpy.ones(positions.shape[:-1]), axis=-1)
        positions = numpy.append(positions, w_array, axis=-1)
    else:
        raise ValueError("Unrecognised form for input args")

    return positions
