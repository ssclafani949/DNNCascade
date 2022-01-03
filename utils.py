import os
import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
import csky as cy


class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


def print_result(title, n_trials, trial, pval, pval_nsigma, add_items={}):
    """Print unblinding results to console

    Parameters
    ----------
    title : str
        The name of the analysis result. Will be displayed as title
        of result box.
    n_trials : int
        The number of background trials on which the p-values are based on.
    trial : tuple
        The trial result.
    pval : float
        The p-value for the given trial.
    pval_nsigma : float
        The p-value in terms of n-sigma for the given trial.
    add_items : dict, optional
        Additional items to print.
    """
    print()
    print('============================================')
    print('=== {}'.format(title))
    print('============================================')
    print('    Number of Background Trials: {}'.format(n_trials))
    print('    TS: {:3.3f}'.format(trial[0]))
    print('    ns: {:3.3f}'.format(trial[1]))
    for key, value in add_items.items():
        print('    {}: {}'.format(key, value))
    print('    p-value: {:3.3e}'.format(pval))
    print('    n-sigma: {:3.2f}'.format(pval_nsigma))

    if pval_nsigma < 3.:
        msg = bcolors.RED + '    --> No significant discovery!'
    elif pval_nsigma < 5.:
        msg = bcolors.YELLOW + '    --> Found evidence for a source!'
    else:
        msg = bcolors.GREEN + '    --> Found a source!'
    msg += bcolors.ENDC
    print(msg)
    print('============================================')
    print()


def plot_skymap(
            skymap, outfile=None, figsize=(9, 6),
            vmin=None, vmax=None, label=None,
        ):
    """Plot a skymap

    Parameters
    ----------
    skymap : array_like
        The skymap to plot.
    outfile : str, optional
        The output file path to which to plot if provided.
    vmin : float, optional
        The minimum value for the colorbar.
    vmax : float, optional
        The maximum value for the colorbar.
    figsize : tuple, optional
        The figure size to use.
    label : str, optional
        The label for the colorbar.

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.
    """
    fig, ax = plt.subplots(
        subplot_kw=dict(projection='aitoff'), figsize=figsize)
    sp = cy.plotting.SkyPlotter(
        pc_kw=dict(cmap=cy.plotting.skymap_cmap, vmin=vmin, vmax=vmax))
    mesh, cb = sp.plot_map(ax, skymap, n_ticks=2)
    kw = dict(color='.5', alpha=.5)
    sp.plot_gp(ax, lw=.5, **kw)
    sp.plot_gc(ax, **kw)
    ax.grid(**kw)
    cb.set_label(label)
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile)

    return fig, ax


def plot_skymap_p_value(scan, **kwargs):
    """Plot a skymap of the TS value

    Parameters
    ----------
    scan : array_like
        The skyscanner scan result.
    **kwargs
        Additional keyword arguments passed on to `plot_skymap()`

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.

    """
    return plot_skymap(
        skymap=scan[0], label=r'$\log_{10}(p_\mathrm{pre-trial})$', **kwargs
    )


def plot_skymap_ts(scan, **kwargs):
    """Plot a skymap of the TS value

    Parameters
    ----------
    scan : array_like
        The skyscanner scan result.
    **kwargs
        Additional keyword arguments passed on to `plot_skymap()`

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.

    """
    return plot_skymap(skymap=scan[1], label=r'TS', **kwargs)


def plot_skymap_ns(scan, **kwargs):
    """Plot a skymap of the ns value

    Parameters
    ----------
    scan : array_like
        The skyscanner scan result.
    **kwargs
        Additional keyword arguments passed on to `plot_skymap()`

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.

    """
    return plot_skymap(skymap=scan[2], label=r'ns', **kwargs)


def plot_skymap_gamma(scan, **kwargs):
    """Plot a skymap of the gamma value

    Parameters
    ----------
    scan : array_like
        The skyscanner scan result.
    **kwargs
        Additional keyword arguments passed on to `plot_skymap()`

    Returns
    -------
    fig, ax
        The matplotlib figure and axis.

    """
    return plot_skymap(skymap=scan[3], label=r'Gamma $\gamma$', **kwargs)


def plot_ss_trial(
            scan, outdir, figsize=(9, 6),
            kwargs_pvalue={'vmin': 0, 'vmax': 5},
            kwargs_ts={'vmin': 0, 'vmax': None},
            kwargs_ns={'vmin': 0, 'vmax': None},
            kwargs_gamma={'vmin': 0., 'vmax': 4.},
        ):
    """Plot a skymap of the gamma value

    Parameters
    ----------
    scan : array_like
        The skyscanner scan result.
    outdir : None, optional
        The output directory to which to plot if provided.
    kwargs_* : dict, optional
        Keyword arguments passed on to plotting functions of
        pvalue, gamma, ns, ts.
    """

    file = os.path.join(outdir, 'skyscan_{}_map.png')

    plot_skymap_p_value(scan, outfile=file.format('pvalue'), **kwargs_pvalue)
    plot_skymap_ts(scan, outfile=file.format('ts'), **kwargs_ts)
    plot_skymap_ns(scan, outfile=file.format('ns'), **kwargs_ns)
    plot_skymap_gamma(scan, outfile=file.format('gamma'), **kwargs_gamma)


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


def recalculate_scan(scan, ts_to_p):
    """Recalculate skyscan based on provided `ts_to_p` function.

    Parameters
    ----------
    scan : array_like
        The scan result of cy.trial.SkyScanTrialRunner.get_one_scan().
        Shape: [4, npix]
    ts_to_p : callable
        The conversion function that maps (dec, ts) --> p-value.

    Returns
    -------
    array_like
        The scan with p-values recalculated based on `ts_to_p`.
        Shape: [4, npix]
    """

    # scan shape: [4, npix]
    # with [-log10(p), ts, ns, gamma] along first axis
    assert len(scan) == 4

    # make a new array, so that we don't modify previous scan
    scan = np.array(scan)

    ts_vals = scan[1]
    nside = hp.get_nside(ts_vals)
    scan_ra, scan_dec = cy.trial.SkyScanner.get_healpix_grid(nside)

    # make sure that decs outside range have ts of 0

    mlog10p = -np.log10(ts_to_p(scan_dec, ts_vals))

    # set new p-values
    scan[0] = mlog10p

    return scan
