import numpy


def transformation_matrix(x_vector, y_vector, translation, spacing):
    """
    Creates a transformation matrix which will convert from a specified
    coordinate system to the scanner frame of reference.

    :param x_vector: The unit vector along the space X axis in scanner coordinates
    :param y_vector: The unit vector along the space Y axis in scanner coordinates
    :param translation: The origin of the space in scanner coordinates
    :param spacing: The size of a space unit in scanner units
    :return:
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
