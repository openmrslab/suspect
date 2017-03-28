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
    spacing : float
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
