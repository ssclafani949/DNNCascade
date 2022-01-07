#!/usr/bin/env python
# encoding: utf-8

"""Script to check statistics of background trials

This script can be used to obtain an overview of the existing background
trails. Based on the given number of background trials, the most significant
p-value is computed that may still be trusted. For p-values above this value,
additional background trials may be necessary.
The boundary p-value is defined as the p-value at which `N` background trials
remain that lead to higher TS values. `N` can be set via the flag
`--trials-after`. Call script via:

    python check_trials.py --trials-after 100
"""

import os
import numpy as np
import click
from scipy import stats
import csky as cy
import datetime

import config as cg
import utils

repo, ana_dir, base_dir, job_basedir = (
    cg.repo, cg.ana_dir, cg.base_dir, cg.job_basedir
)
now = datetime.datetime.now


@click.command()
@click.option('-a', '--ana', 'ana_name', default='DNNC', help='Dataset title')
@click.option('--ana-dir', default=ana_dir, type=click.Path())
@click.option('--job_basedir', default=job_basedir, type=click.Path())
@click.option('--save/--nosave', default=False)
@click.option('--base-dir', default=base_dir,
              type=click.Path(file_okay=False, writable=True))
@click.option('--trials-after', default=100, type=int)
@click.option('--nside', default=128, type=int)
@click.option(
    '--check', '-c', multiple=True,
    default=['sourcelist', 'gp', 'stacking', 'skyscan', 'fermibubbles'])
def main(ana_name, ana_dir, job_basedir, save, base_dir, trials_after, nside,
         check):
    """Check background trials.
    """

    # setup ana
    state = State(ana_name, ana_dir, save, base_dir, job_basedir)

    check = [str(c).lower() for c in check]

    check_options = ['sourcelist', 'gp', 'stacking', 'skyscan', 'fermibubbles']
    for c in check:
        if c not in check_options:
            raise KeyError('Unkown analysis: {}, options are {}'.format(
                c, check_options))

    # -----------
    # Source List
    # -----------
    if 'sourcelist' in check:
        check_sourcelist(state, trials_after=trials_after)

    # --------------
    # Galactic Plane
    # --------------
    if 'gp' in check:
        check_gp(state, trials_after=trials_after)

    # -------------
    # Fermi Bubbles
    # -------------
    if 'fermibubbles' in check:
        check_fermibubbles(state, trials_after=trials_after)

    # ----------------
    # Catalog stacking
    # ----------------
    if 'stacking' in check:
        check_stacking(state, trials_after=trials_after)

    # --------------
    # All-sky search
    # --------------
    if 'skyscan' in check:
        check_skyscan(state, trials_after=trials_after, nside=nside)


class State(object):
    def __init__(self, ana_name, ana_dir, save, base_dir, job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = (
            ana_name, ana_dir, save, job_basedir
        )
        self.base_dir = base_dir


def pval_to_nsigma(pvalue):
    """Converta  p-value to number of sigma assuming a normal distribution.

    Parameters
    ----------
    pvalue : array_like
        The p-value to convert.

    Returns
    -------
    array_like
        The number of sigma that the p-value corresponds to for a
        normal distribution.
    """
    return stats.norm.isf(pvalue)


def get_pval_bound(bg, trials_after=100):
    """Get boundary up-to which p-value can be trusted given the bkg trials

    Parameters
    ----------
    bg : array_like or cy.dist.TSD
        The background trials
    trials_after : int, optional
        The desired number of trials after the boundary p-value.
        A p-value is considered as trustworthy given the amount of background
        trials, if there are at least `trials_after` many background trials
        with a higher TS value than the boundary p-value.
        Note: this is also the factor of additional stats compared to the
        necessary number of trials to make a p-value determination of
        1/n_trials.

    Returns
    -------
    TYPE
        Description
    """
    n_trials = float(len(bg))
    return trials_after/n_trials


def print_msg(msg, nsigma):
    """Print colored message depending on nsgima.

    Parameters
    ----------
    msg : str
        The message to print.
    nsigma : float
        The boundary p-value in term of n-sigma.
    """
    if nsigma < 3:
        print(utils.bcolors.RED + msg + utils.bcolors.ENDC)
    elif nsigma < 5:
        print(utils.bcolors.YELLOW + msg + utils.bcolors.ENDC)
    else:
        print(msg)


def check_sourcelist(state, trials_after=100):
    """Check source list background trials

    Parameters
    ----------
    state : State
        A collection of settings.
    trials_after : int, optional
        The desired number of trials after the boundary p-value.
        A p-value is considered as trustworthy given the amount of background
        trials, if there are at least `trials_after` many background trials
        with a higher TS value than the boundary p-value.
        Note: this is also the factor of additional stats compared to the
        necessary number of trials to make a p-value determination of
        1/n_trials.
    """
    base_dir = os.path.join(state.base_dir, 'ps/correlated_trials')
    bgfile_corr = '{}/correlated_bg.npy'.format(base_dir)
    trials_corr = np.load(bgfile_corr, allow_pickle=True)
    bg_corr = cy.dists.TSD(trials_corr)

    bgfile = '{}/pretrial_bgs.npy'.format(base_dir)
    bgs = np.load(bgfile, allow_pickle=True)

    src_list_file = os.path.join(cg.catalog_dir, 'Source_List_DNNC.npy')
    sourcelist = np.load(src_list_file, allow_pickle=True)

    print()
    print('====================================================')
    print('=== Checking bkg trials for Source-list')
    print('====================================================')
    pval_max = -np.inf
    nsigma_min = None
    n_min = None

    # loop through each source
    for i, source in enumerate(sourcelist):
        pval = get_pval_bound(bgs[i], trials_after=trials_after)
        nsigma = pval_to_nsigma(pval)

        if pval > pval_max:
            pval_max = pval
            nsigma_min = nsigma
            n_min = len(bgs[i])

        msg = '   #{:03d} | n-trials: {:7d} | min. p-value: {:3.3e} '
        msg += ' | max. n-sigma: {:5.2f} | {}'
        msg = msg.format(i, len(bgs[i]), pval, nsigma, source[0])
        print_msg(msg, nsigma)

    print()
    print('Hottest source:')
    print('---------------')

    msg = '   [{:<10}] n-trials: {:7d} | min. p-value: {:3.3e} | '
    msg += 'max. n-sigma: {:5.2f}'
    print_msg(msg.format('pre-trial', n_min, pval_max, nsigma_min), nsigma_min)

    pval = get_pval_bound(bg_corr, trials_after=trials_after)
    nsigma = pval_to_nsigma(pval)
    print_msg(msg.format('post-trial', len(bg_corr), pval, nsigma), nsigma)


def check_gp(state, trials_after=100):
    """Check background trials for galactic plane

    Parameters
    ----------
    state : State
        A collection of settings.
    trials_after : int, optional
        The desired number of trials after the boundary p-value.
        A p-value is considered as trustworthy given the amount of background
        trials, if there are at least `trials_after` many background trials
        with a higher TS value than the boundary p-value.
        Note: this is also the factor of additional stats compared to the
        necessary number of trials to make a p-value determination of
        1/n_trials.
    """
    print()
    print('====================================================')
    print('=== Checking bkg trials for Galactic Plane templates')
    print('====================================================')

    # loop through each catalog
    for template in ['kra5', 'kra50', 'pi0']:

        base_dir = state.base_dir + '/gp/trials/{}/{}/'.format(
            state.ana_name, template)
        sigfile = '{}/trials.dict'.format(base_dir)
        sig = np.load(sigfile, allow_pickle=True)
        bg = cy.dists.TSD(sig['poisson']['nsig'][0.0]['ts'])

        pval = get_pval_bound(bg, trials_after=trials_after)
        nsigma = pval_to_nsigma(pval)
        msg = '   n-trials: {:7d} | min. p-value: {:3.3e} | '
        msg += 'max. n-sigma: {:5.2f} | template: {}'
        print_msg(msg.format(len(bg), pval, nsigma, template), nsigma)
    print('====================================================')


def check_fermibubbles(state, trials_after=100):
    """Check background trials for Fermi bubbles

    Parameters
    ----------
    state : State
        A collection of settings.
    trials_after : int, optional
        The desired number of trials after the boundary p-value.
        A p-value is considered as trustworthy given the amount of background
        trials, if there are at least `trials_after` many background trials
        with a higher TS value than the boundary p-value.
        Note: this is also the factor of additional stats compared to the
        necessary number of trials to make a p-value determination of
        1/n_trials.
    """
    print()
    print('====================================================')
    print('=== Checking bkg trials for Fermi bubbles templates')
    print('====================================================')

    base_dir = state.base_dir + '/gp/trials/{}/fermibubbles/'.format(
        state.ana_name)
    bgfile_corr = '{}/correlated_trials/correlated_bg.npy'.format(base_dir)
    trials_corr = np.load(bgfile_corr, allow_pickle=True)
    bg_corr = cy.dists.TSD(trials_corr)

    # load bkg trials for each cutoff
    sig = np.load('{}/trials.dict'.format(base_dir), allow_pickle=True)
    pval_max = -np.inf
    nsigma_min = None
    n_min = None

    # loop through cutoffs
    for cutoff in [50, 100, 500, np.inf]:
        bg = cy.dists.TSD(sig['poisson']['cutoff'][cutoff]['nsig'][0.0]['ts'])

        pval = get_pval_bound(bg, trials_after=trials_after)
        nsigma = pval_to_nsigma(pval)
        if pval > pval_max:
            pval_max = pval
            nsigma_min = nsigma
            n_min = len(bg)

        msg = '   n-trials: {:7d} | min. p-value: {:3.3e} '
        msg += ' | max. n-sigma: {:5.2f} | cutoff: {:3.3f} TeV'
        print_msg(msg.format(len(bg), pval, nsigma, cutoff), nsigma)

    print()
    print('Most significant cutoff:')
    print('------------------------')

    msg = '   [{:<10}] n-trials: {:7d} | min. p-value: {:3.3e} | '
    msg += 'max. n-sigma: {:5.2f}'
    print_msg(msg.format('pre-trial', n_min, pval_max, nsigma_min), nsigma_min)

    pval = get_pval_bound(bg_corr, trials_after=trials_after)
    nsigma = pval_to_nsigma(pval)
    print_msg(msg.format('post-trial', len(bg_corr), pval, nsigma), nsigma)


def check_stacking(state, trials_after=100):
    """Check background trials for stacking searches

    Parameters
    ----------
    state : State
        A collection of settings.
    trials_after : int, optional
        The desired number of trials after the boundary p-value.
        A p-value is considered as trustworthy given the amount of background
        trials, if there are at least `trials_after` many background trials
        with a higher TS value than the boundary p-value.
        Note: this is also the factor of additional stats compared to the
        necessary number of trials to make a p-value determination of
        1/n_trials.
    """
    print()
    print('====================================================')
    print('=== Checking bkg trials for catalog stacking')
    print('====================================================')
    for catalog in ['snr', 'pwn', 'unid']:

        # load trials
        bgfile = '{}/stacking//{}_bg.dict'.format(state.base_dir, catalog)
        b = np.load(bgfile, allow_pickle=True)
        bg = cy.dists.TSD(b)

        pval = get_pval_bound(bg, trials_after=trials_after)
        nsigma = pval_to_nsigma(pval)
        msg = '   n-trials: {:7d} | min. p-value: {:3.3e} '
        msg += ' | max. n-sigma: {:5.2f} | catalog: {}'
        print_msg(msg.format(len(bg), pval, nsigma, catalog), nsigma)


def check_skyscan(state, trials_after=100, nside=128):
    """Check background trials for sky-scan

    Parameters
    ----------
    state : State
        A collection of settings.
    trials_after : int, optional
        The desired number of trials after the boundary p-value.
        A p-value is considered as trustworthy given the amount of background
        trials, if there are at least `trials_after` many background trials
        with a higher TS value than the boundary p-value.
        Note: this is also the factor of additional stats compared to the
        necessary number of trials to make a p-value determination of
        1/n_trials.
    nside : int, optional
        The healpix nside to use for the skyscan.

    Raises
    ------
    ValueError
        If background trials don't exist for certain declinations, or if
        these aren't sampled dense enough.
    """
    print()
    print('====================================================')
    print('=== Checking bkg trials for all-sky search')
    print('====================================================')

    # load uncorrelated bg trials at each declination
    base_dir = state.base_dir + '/ps/trials/DNNC'
    bgfile = '{}/bg.dict'.format(base_dir)
    bgs = np.load(bgfile, allow_pickle=True)['dec']

    # load correlated sky scan trials for trial-correction
    trials_corr_file = state.base_dir + '/skyscan/trials/DNNC/sky_scan_bg.npy'
    trials_corr_dict = np.load(trials_corr_file, allow_pickle=True)
    trials_corr = trials_corr_dict['nside'][nside]['bg']['nofit']

    assert len(trials_corr.mlog10p_north) == len(trials_corr.mlog10p_north)
    bg_corr = trials_corr.mlog10p_north

    # check if boundary declination values exist
    for k in [-81., 81.]:
        if k not in bgs:
            raise ValueError('Declination: {}° does not exist in {}!'.format(
                k, list(bgs.keys())))

    # check if any larger gaps than 2° exist in declination
    gaps = np.diff(list(bgs.keys()))
    if (gaps > 2. + 1e-7).any():
        msg = 'Found missing background trials for PS declinations. '
        msg += 'Found a gap with: {}°! Here are all existing declinations: {}.'
        raise ValueError(msg.format(np.max(gaps)), list(bgs.keys()))

    pval_max = -np.inf
    nsigma_min = None
    n_min = None
    for dec_deg in bgs.keys():
        bg = bgs[dec_deg]

        pval = get_pval_bound(bg, trials_after=trials_after)
        nsigma = pval_to_nsigma(pval)

        if pval > pval_max:
            pval_max = pval
            nsigma_min = nsigma
            n_min = len(bg)

        msg = '   n-trials: {:7d} | min. p-value: {:3.3e} '
        msg += ' | max. n-sigma: {:5.2f} | nside: {} | dec: {:3.2f}°'
        print_msg(msg.format(len(bg), pval, nsigma, nside, dec_deg), nsigma)

    print()
    print('Most significant pixel:')
    print('------------------------')

    msg = '   [{:<10}] n-trials: {:7d} | min. p-value: {:3.3e} | '
    msg += 'max. n-sigma: {:5.2f}'
    print_msg(msg.format('pre-trial', n_min, pval_max, nsigma_min), nsigma_min)

    pval = get_pval_bound(bg_corr, trials_after=trials_after)
    nsigma = pval_to_nsigma(pval)
    print_msg(msg.format('post-trial', len(bg_corr), pval, nsigma), nsigma)


if __name__ == '__main__':
    main()
