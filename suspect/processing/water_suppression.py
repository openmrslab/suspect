import numpy

import suspect.basis


def hsvd(data, rank, L=None):
    """
    Parameters
    ----------
    data :
    rank :
    L :

    Returns
    -------

    Todo
    ----

    """
    if L is None:
        L = data.np // 2
    # start by building the Hankel matrix
    hankel_matrix = numpy.zeros((L, data.np - L), "complex")
    for i in range(int(data.np - L)):
        hankel_matrix[:, i] = data[i:(i + L)]

    # perform the singular value decomposition
    U, s, V = numpy.linalg.svd(numpy.matrix(hankel_matrix))
    V = V.H  # numpy returns the Hermitian conjugate of V

    # truncate the matrices to the given rank
    U_K = U[:, :rank]
    V_K = V[:, :rank]
    s_K = numpy.matrix(numpy.diag(s[:rank]))

    # because of the structure of the Hankel matrix, each row of U_K is the
    # result of multiplying the previous row by the delta t propagator matrix
    # Z' (a similar result holds for V as well). This gives us U_Kb * Z' = U_Kt
    # where U_Kb is U_K without the bottom row and U_Kt is U_K without the top
    # row.
    U_Kt = U_K[1:, :]
    U_Kb = U_K[:-1, :]
    # this gives us a set of linear equations which can be solved to find Z'.
    # Because of the noise in the system we solve with least-squares
    Zp = numpy.linalg.inv(U_Kb.H * U_Kb) * U_Kb.H * U_Kt

    # in the right basis, Zp is just the diagonal matrix describing the
    # evolution of each frequency component, by diagonalising the matrix we can
    # find that basis and get the z = exp((-damping + j*2pi * f) * dt) terms

    # alternatively we can just get the eigenvalues instead
    val, vec = numpy.linalg.eig(Zp)

    # the magnitude gives the damping and the angle gives the frequency
    damping_coeffs = numpy.zeros(rank)
    frequency_coeffs = numpy.zeros(rank)
    for i in range(rank):
        damping_coeffs[i] = - numpy.log(abs(val[i])) / data.dt
        frequency_coeffs[i] = numpy.angle(val[i]) / (data.dt * 2 * numpy.pi)

    # TODO in theory we can calculate the magnitude of each signal from the
    # RHS decomposition, linalg.inv(vec) * (S_K * V_K.H)[:, 0]

    # a simpler but more expensive way is to construct a basis set from the
    # known damping and frequency components and fit to the original data to
    # get the amplitudes and phase data
    X = numpy.zeros((data.np, rank), "complex")
    # TODO this should use the singlet fitting module to make the basis
    for i in range(rank):
        X[:, i] = suspect.basis.lorentzian(data.time_axis(),
                                           frequency_coeffs[i],
                                           0,
                                           damping_coeffs[i] / numpy.pi) * data.np

    # we use the linear non-iterative least squares again
    U2, s2, V2 = numpy.linalg.svd(numpy.matrix(X), full_matrices=False)
    s2_inv = numpy.diag(1 / s2)
    beta = V2.H * s2_inv * U2.H * numpy.matrix(numpy.reshape(data, (data.np, 1)))

    components = []
    for i in range(rank):
        components.append({
            "amplitude": float(abs(beta[i])),
            "phase": float(numpy.angle(beta[i])),
            "fwhm": damping_coeffs[i] / numpy.pi,
            "frequency": frequency_coeffs[i]
        })

    return components


def construct_fid(components, time_axis):
    fid = numpy.zeros_like(time_axis, 'complex')
    for i in range(len(components)):
        lorentzian = suspect.basis.lorentzian(time_axis,
                                              components[i]["frequency"],
                                              components[i]["phase"],
                                              components[i]["fwhm"])
        fid += components[i]["amplitude"] * lorentzian * len(time_axis)
    return fid
