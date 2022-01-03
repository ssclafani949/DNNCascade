import numpy as np
import healpy as hp
import csky as cy


def get_mask_north_dict(nside_list):
    """Get a mask for healpix pixels belonging to the northern hemisphere

    Parameters
    ----------
    nside_list : list of int
        The list of nside for which to compute the mask

    Returns
    -------
    dict
        A dictionary containing the mask for each nside from `nside_list`.
    """
    mask_north_dict = {}
    for nside in nside_list:
        theta, phi = hp.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))
        mask_north_dict[nside] = theta <= np.pi/2.

    return mask_north_dict


def extract_hottest_p_value(ss_trial, mask_north_dict=None):
    """Get hottest p-value from sky scan trial

    Parameters
    ----------
    ss_trial : array_like
        The sky scan result from the skyscanner for a single trial.
    mask_north_dict : None, optional
        A dictionary with key value pairs: {nside: mask} where mask is a
        boolean mask that indicates which healpix pixels belong to the
        northern hemisphere.

    Returns
    -------
    cy.utils.Arrays
        The maximum p-values for the entire sky, northern (dec >= 0) sky
        and southern (dec < 0) sky.
        Shape: (3,)
    """

    # ss_trial shape: [4, npix]
    # with [-log10(p), ts, ns, gamma] along first axis
    mlog10ps_sky = ss_trial[0]

    # get mask for northern/southern pixels
    nside = hp.get_nside(mlog10ps_sky)
    if mask_north_dict is None:
        mask_north_dict = get_mask_north_dict([nside])
    mask_north = mask_north_dict[nside]

    mlog10p_allsky = np.max(mlog10ps_sky)
    mlog10p_north = np.max(mlog10ps_sky[mask_north])
    mlog10p_south = np.max(mlog10ps_sky[~mask_north])

    return cy.utils.Arrays({
        'mlog10p_allsky': mlog10p_allsky,
        'mlog10p_north': mlog10p_north,
        'mlog10p_south': mlog10p_south,
    })
