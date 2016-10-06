import numpy as np


def complex_array_from_iter(data_iter, length=-1, shape=None, chirality=1):
    """
    Converts an iterable over a series of real, imaginary pairs into
    a numpy.ndarray of complex64.

    Parameters
    ----------
    data_iter : iter
        The iterator over the points
    length : int
        The number of complex points to read. The defaults is -1, which
        means all data is read.
    shape : array-like
        Shape to convert the final array to. The array will be squeezed
        after reshaping to remove any dimensions of length 1.

    Returns
    -------
    out : ndarray
        The output array
    """
    complex_iter = (complex(r, chirality * i) for r, i in zip(data_iter, data_iter))
    complex_array = np.fromiter(complex_iter, "complex64", length)
    if shape is not None:
        complex_array = np.reshape(complex_array, shape).squeeze()
    return complex_array
