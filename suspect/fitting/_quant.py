import numpy as np

water_molal = 55.51e3
water_molar = 55.01e3


def attenuation_scaling_factor(t1, t2, te, tr):
    """
    Calculates the expected attenuation in an MRS signal arising due to T1 and
    T2 effects, according to the equation:
    :math:`\mathrm{e}^{-\\frac{TE}{T2}}(1 - \mathrm{e}^{-\\frac{TR}{T1}})`

    Parameters
    ----------
    t1
        The T1 of the tissue
    t2
        The T2 of the tissue
    te
        The echo time of the sequence
    tr
        The repetition time of the sequence

    Returns
    -------
    float
        The attenuation factor in the observed signal expected due to T1 and T2
        effects

    Notes
    -----
    This function uses the simplified form of the equation, assuming that TE
    is much shorter than T1 and so the sequence behaves roughly as a 90 pulse
    followed by a readout, any refocussing pulses do not substantially change
    the longitudinal magnetisation. If this is not the case then the details
    of the sequence will be important and a more precise form of this
    calculation will be required.
    """
    return np.exp(-te / t2) * (1 - np.exp(-tr / t1))


def molar_concentration_factor(f_wm, f_gm, f_csf, te, tr, tissue_params=None):
    """
    Calculate the scaling factor necessary to obtain molar metabolite
    concentrations from a ratio of metabolite to water peak amplitude.

    Parameters
    ----------
    f_wm : float
        The fraction of the voxel containing white matter
    f_gm : float
        The fraction of the voxel containing grey matter
    f_csf : float
        The fraction of the voxel containing CSF
    te : float
        The echo time of the sequence
    tr : float
        The repetition time of the sequence
    tissue_params : dict, optional
        User supplied values for tissue MR properties including T1 and T2

    Returns
    -------
    float
        The scaling factor to convert metabolite to water ratios to molar concentrations.

    Notes
    -----
    The calculation used here follows the form derived in [1]_.

    There are various parameters required for these calculations which can vary, for
    example with field strength or other conditions. Default values are provided in
    Suspect (tissue water concentrations are drawn from Gasparovic 2006, relaxation
    times from Gussew 2012), but they can be overridden by passing in alternative values in the
    tissue_parameters argument. The relevant parameters are:

    ========== ============================================ =============
    Key        Description                                  Default Value
    ---------- -------------------------------------------- -------------
    beta_wm    the water density of white matter            0.65
    beta_gm    the water density of grey matter             0.78
    beta_csf   the water density of CSF                     0.97
    h2o_t1_wm  the T1 of water in white matter              1080ms
    h2o_t1_gm  the T1 of water in grey matter               1820ms
    h2o_t1_csf the T1 of water in CSF                       4160ms
    h2o_t2_wm  the T2 of water in white matter              70ms
    h2o_t2_gm  the T2 of water in grey matter               100ms
    h2o_t2_csf the T2 of water in CSF                       500ms
    met_t1     the T1 of metabolites in brain tissue        1400ms
    met_t2     the T2 of metabolites in brain tissue        200ms
    ========== ============================================ =============

    References
    ----------
    .. [1] Near, J., Harris, A. D., Juchem, C., Kreis, R., Marjańska, M., Öz, G., et al. (2020). Preprocessing, analysis and quantification in single‐voxel magnetic resonance spectroscopy: experts' consensus recommendations. NMR in Biomedicine, 29, 323–23. http://doi.org/10.1002/nbm.4257
    """
    tissue_parameters = {
        "beta_wm": 0.65,
        "beta_gm": 0.78,
        "beta_csf": 0.97,
        "h2o_t1_wm": 1080,
        "h2o_t1_gm": 1820,
        "h2o_t1_csf": 4160,
        "h2o_t2_wm": 70,
        "h2o_t2_gm": 100,
        "h2o_t2_csf": 500,
        "met_t1": 1400,
        "met_t2": 200
    }
    if tissue_params is not None:
        tissue_parameters.update(tissue_params)

    r_wm = attenuation_scaling_factor(tissue_parameters["h2o_t1_wm"],
                                      tissue_parameters["h2o_t2_wm"],
                                      te, tr)
    r_gm = attenuation_scaling_factor(tissue_parameters["h2o_t1_gm"],
                                      tissue_parameters["h2o_t2_gm"],
                                      te, tr)
    r_csf = attenuation_scaling_factor(tissue_parameters["h2o_t1_csf"],
                                       tissue_parameters["h2o_t2_csf"],
                                       te, tr)
    r_met = attenuation_scaling_factor(tissue_parameters["met_t1"],
                                       tissue_parameters["met_t2"],
                                       te, tr)

    return (f_wm * tissue_parameters["beta_wm"] * r_wm +
            f_gm * tissue_parameters["beta_gm"] * r_gm +
            f_csf * tissue_parameters["beta_csf"] * r_csf) * \
        water_molar / r_met / (1 - f_csf)