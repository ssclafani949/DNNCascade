#!/usr/bin/env python

import csky as cy
import numpy as np
import healpy as hp
import pickle
import datetime
import socket
import histlite as hl
import astropy
import matplotlib.pyplot as plt
import click
import sys
import os
import time
import config as cg
now = datetime.datetime.now
flush = sys.stdout.flush
hp.disable_warnings()

repo, ana_dir, base_dir, job_basedir = (
    cg.repo, cg.ana_dir, cg.base_dir, cg.job_basedir
)


class State (object):
    def __init__(self, ana_name, ana_dir, save, base_dir, job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = (
            ana_name, ana_dir, save, job_basedir)
        self.base_dir = base_dir
        self._ana = None

    @property
    def ana(self):
        if self._ana is None:
            repo.clear_cache()
            if 'baseline' in base_dir:
                specs = cy.selections.DNNCascadeDataSpecs.DNNC_10yr
            elif 'systematics' in base_dir:
                specs = cy.selections.DNNCascadeDataSpecs.DNNC_10yr_systematics
            ana = cy.get_analysis(repo, 'version-001-p00', specs, dir=base_dir)
            if self.save:
                cy.utils.ensure_dir(self.ana_dir)
                ana.save(self.ana_dir)
            ana.name = self.ana_name
            self._ana = ana
        return self._ana

    @property
    def state_args(self):
        return '--ana {} --ana-dir {} --base-dir {}'.format(
            self.ana_name, self.ana_dir, self.base_dir)


pass_state = click.make_pass_decorator(State)


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
    print('    p-value: {}'.format(pval))
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


@click.group(invoke_without_command=True, chain=True)
@click.option('-a', '--ana', 'ana_name', default='DNNC', help='Dataset title')
@click.option('--ana-dir', default=ana_dir, type=click.Path())
@click.option('--job_basedir', default=job_basedir, type=click.Path())
@click.option('--save/--nosave', default=False)
@click.option('--base-dir', default=base_dir,
              type=click.Path(file_okay=False, writable=True))
@click.pass_context
def cli(ctx, ana_name, ana_dir, save, base_dir, job_basedir):
    ctx.obj = State.state = State(
        ana_name, ana_dir, save, base_dir, job_basedir)


@cli.resultcallback()
def report_timing(result, **kw):
    exe_t1 = now()
    print ('c11: end at {} .'.format(exe_t1))
    print ('c11: {} elapsed.'.format(exe_t1 - exe_t0))


@cli.command()
@pass_state
def setup_ana(state):
    state.ana


@cli.command()
@click.option('--seed', default=None, type=int, help='Trial injection seed')
@click.option('--TRUTH', default=None, type=bool,
              help='Must be Set to TRUE to unblind')
@pass_state
def unblind_sourcelist(
        state, seed, truth,  logging=True):
    """
    Unblind Source List
    """
    trials = []
    sourcelist = np.load('catalogs/Source_List_DNNC.npy', allow_pickle=True)
    t0 = now()
    print(truth)
    ana = state.ana
    dir = cy.utils.ensure_dir('{}/ps/'.format(state.base_dir))

    # load correlated MultiTrialRunner background trials
    print('Loading correlated background trials...')
    base_dir = os.path.join(state.base_dir, 'ps/correlated_trials')
    bgfile_corr = '{}/correlated_bg.npy'.format(base_dir)
    trials_corr = np.load(bgfile_corr, allow_pickle=True)
    bg_corr = cy.dists.TSD(trials_corr)

    # load bkg trials at source declinations
    print('Loading background trials at source declinations...')
    bgfile = '{}/pretrial_bgs.npy'.format(base_dir)
    bgs = np.load(bgfile, allow_pickle=True)
    assert np.alltrue([isinstance(bg, cy.dists.TSD) for bg in bgs])
    assert len(bgs) == len(sourcelist)

    def get_tr(dec, ra,  truth):
        src = cy.utils.sources(ra, dec, deg=False)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf=conf, TRUTH=truth)
        return tr, src

    for i, source in enumerate(sourcelist):

        tr, src = get_tr(
            dec=np.radians(source[2]),
            ra=np.radians(source[1]),
            truth=truth)

        if truth:
            print('UNBLINDING!!!')

        trial = tr.get_one_fit(TRUTH=truth, seed=seed, logging=logging)

        # compute pre-trial p-value for given source
        bg = bgs[i]
        pval = bg.sf(trial[0], fit=False)
        pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
        trial.append(pval)
        trial.append(pval_nsigma)

        # print result
        msg = '#{:03d} | ts: {:5.2f} | ns: {:6.2f} | gamma: {:4.2f}'
        msg += ' | n-sigma: {:5.2f} | {}'
        msg = msg.format(i, *trial[:3], pval_nsigma, source[0])

        if pval_nsigma < 3.:
            print(msg)
        elif pval_nsigma < 5.:
            print(bcolors.YELLOW + msg + bcolors.ENDC)
        else:
            print(bcolors.RED + msg + bcolors.ENDC)

        # append results of this source to overall trials
        trials.append(trial)

    # get hottest source
    min_idx = np.argmin(np.array(trials)[:, 3])
    min_source = sourcelist[min_idx][0]
    pval_min = trials[min_idx][3]

    # compute trial-corrected p-value
    ts_mlog10p = -np.log10(pval_min)
    pval = bg_corr.sf(ts_mlog10p, fit=False)
    pval_nsigma = bg.sf_nsigma(ts_mlog10p, fit=False)

    # print results to console
    print_result(
        title='Results for source list',
        n_trials=len(bg), trial=trial, pval=pval, pval_nsigma=pval_nsigma,
        add_items={
            'hottest source': min_source,
            'pre-trial p-value': pval_min,
        },
    )

    flush()
    t1 = now()
    if truth:
        out_dir = cy.utils.ensure_dir('{}/ps/results/'.format(
            state.base_dir, state.ana_name))
        out_file = '{}/fulllist_unblinded.npy'.format(
            out_dir)
        print ('-> {}'.format(out_file))
        np.save(out_file, (trials, pval, pval_nsigma, min_idx, min_source))


@cli.command()
@click.argument('temp')
@click.option('--seed', default=None, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--TRUTH', default=None, type=bool,
              help='Must be Set to TRUE to unblind')
@click.option('-c', '--cutoff', default=np.inf, type=float,
              help='exponential cutoff energy (TeV)')
@pass_state
def unblind_gp(
            state, temp,
            seed, cpus,
            truth, cutoff, logging=True):
    """
    Unblind a particular galactic plane templaet
    """
    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    ana = state.ana
    cutoff_GeV = cutoff * 1e3

    if temp == 'fermibubbles' and cutoff not in [50, 100, 500]:
        raise ValueError(
            'Fermibubbles are only being unblinded for cutoffs 50/100/500 TeV,'
            + ' but not for "{:3.3f} TeV"!'.format(cutoff)
        )

    def get_tr(template_str, TRUTH):
        gp_conf = cg.get_gp_conf(
            template_str=template_str,
            cutoff_GeV=cutoff_GeV,
            base_dir=state.base_dir
        )
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr = get_tr(temp, TRUTH=truth)
    t0 = now()
    if truth:
        print('UNBLINDING!!!')

    print('Loading BKG TRIALS')
    base_dir = state.base_dir + '/gp/trials/{}/{}/'.format(
        state.ana_name, temp)
    sigfile = '{}/trials.dict'.format(base_dir)
    sig = np.load(sigfile, allow_pickle=True)
    if temp == 'fermibubbles':

        # safety check to make sure the correct bg trials exist
        # ToDo: csky.bk.get_best should be modified to check for maximum
        # allowed difference, if provided.
        c_keys = list(sig['poisson']['cutoff'].keys())
        if not np.any(np.abs([c - cutoff for c in c_keys]) < 1e-3):
            msg = 'Cutoff {:3.4f} does not exist in bg trials: {}!'
            raise ValueError(msg.format(cutoff, c_keys))

        sig_trials = cy.bk.get_best(sig, 'poisson', 'cutoff', cutoff, 'nsig')
    else:
        sig_trials = sig['poisson']['nsig']
    b = sig_trials[0.0]['ts']
    bg = cy.dists.TSD(b)
    trial = tr.get_one_fit(TRUTH=truth,  seed=seed, logging=logging)
    pval = bg.sf(trial[0], fit=False)
    pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
    trial.append(pval)
    trial.append(pval_nsigma)

    # print results to console
    print_result(
        title='Results for GP template: {}'.format(temp),
        n_trials=len(bg), trial=trial, pval=pval, pval_nsigma=pval_nsigma,
    )
    flush()

    if truth:
        if temp == 'fermibubbles':
            out_dir = cy.utils.ensure_dir('{}/gp/results/{}/'.format(
                state.base_dir, temp))
            out_file = '{}/{}_cutoff_{}_unblinded.npy'.format(
                out_dir, temp, cutoff)
        else:
            out_dir = cy.utils.ensure_dir('{}/gp/results/{}/'.format(
                state.base_dir, temp))
            out_file = '{}/{}_unblinded.npy'.format(
                out_dir, temp)
        print ('-> {}'.format(out_file))
        np.save(out_file, trials)


@cli.command()
@click.option('--TRUTH', default=None, type=bool,
              help='Must be Set to TRUE to unblind')
@click.option('-c', '--cutoff', default=np.inf, type=float,
              help='exponential cutoff energy (TeV)')
@click.option('--seed', default=None, type=int)
@pass_state
def unblind_stacking(state, truth, cutoff, seed, logging=True):
    """
    Unblind all the stacking catalogs
    """
    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print(seed)
    ana = state.ana

    def get_tr(src, TRUTH):
        conf = cg.get_ps_conf(src=src, gamma=2.0, cutoff_GeV=np.inf)
        tr = cy.get_trial_runner(ana=ana, conf=conf, TRUTH=TRUTH)
        return tr
    if truth:
        print('UNBLINDING!!!')
    for catalog in ['snr', 'pwn', 'unid']:
        print('Catalog: {}'.format(catalog))
        cat = np.load('catalogs/{}_ESTES_12.pickle'.format(
            catalog), allow_pickle=True)
        src = cy.utils.Sources(dec=cat['dec_deg'], ra=cat['ra_deg'], deg=True)
        tr = get_tr(src, TRUTH=truth)

        print('Loading BKG TRIALS')
        base_dir = state.base_dir + '/stacking/'
        bgfile = '{}/{}_bg.dict'.format(base_dir, catalog)
        b = np.load(bgfile, allow_pickle=True)
        bg = cy.dists.TSD(b)
        trial = tr.get_one_fit(TRUTH=truth,  seed=seed, logging=logging)
        pval = bg.sf(trial[0], fit=False)
        pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
        trial.append(pval)
        trial.append(pval_nsigma)

        # print results to console
        print_result(
            title='Results for stacking catalog: {}'.format(catalog),
            n_trials=len(bg), trial=trial, pval=pval, pval_nsigma=pval_nsigma,
        )
        flush()

        if truth:
            out_dir = cy.utils.ensure_dir('{}/stacking/results/{}/'.format(
                state.base_dir, catalog))
            out_file = '{}/{}_unblinded.npy'.format(
                out_dir, catalog)
            print ('-> {}'.format(out_file))
            np.save(out_file, trials)


@cli.command()
@click.option('--nside', default=128, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--seed', default=None, type=int)
@click.option('--fit/--nofit', default=False, help='Chi2 Fit or Not')
@click.option('--TRUTH', default=None, type=bool,
              help='Must be Set to TRUE to unblind')
@pass_state
def unblind_skyscan(state, nside, cpus, seed, fit, truth):
    """
    Unblind the skyscan and save the true map
    """

    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    base_dir = state.base_dir + '/ps/trials/DNNC'
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
    bg = np.load(bgfile, allow_pickle=True)
    ts_to_p = lambda dec, ts: cy.dists.ts_to_p(bg['dec'], np.degrees(dec), ts, fit=fit)
    t0 = now()
    ana = state.ana
    conf = cg.get_ps_conf(src=None, gamma=2.0)
    conf.pop('src')
    conf.update({
        'ana': ana,
        'mp_cpus': cpus,
        'extra_keep': ['energy'],
    })
    sstr = cy.get_sky_scan_trial_runner(conf=conf, TRUTH=truth,
                                        min_dec=np.radians(-80),
                                        max_dec=np.radians(80),
                                        mp_scan_cpus=cpus,
                                        nside=nside, ts_to_p=ts_to_p)
    if truth:
        print('UNBLINDING!!!!')
    trials = sstr.get_one_scan(logging=True, seed=seed, TRUTH=truth)
    if truth:
        out_dir = cy.utils.ensure_dir('{}/skyscan/results/'.format(
            state.base_dir))
        out_file = '{}/unblinded_skyscan.npy'.format(
            out_dir,  seed)
        print ('-> {}'.format(out_file))
        np.save(out_file, trials)


if __name__ == '__main__':
    exe_t0 = now()
    print ('start at {} .'.format(exe_t0))
    cli()
