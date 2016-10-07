from .mrsdata import MRSData


def adjust_phase(data, zero_phase, first_phase=0, fixed_frequency=0):
    """
    Adjust the phase of an MRSData object

    Parameters
    ----------
    data : MRSData
        The MRSData object to be phased
    zero_phase : scalar
        The change to the zero order phase, in radians
    first_phase : scalar, optional
        The change to the first order phase, in radians per Hz
    fixed_frequency : scalar, optional
        The frequency, in Hz, which is unchanged by the first order
        phase shift

    Returns
    -------
    out : MRSData
        A new MRSData object with adjusted phase.
    """
    return data.adjust_phase(zero_phase, first_phase, fixed_frequency)
