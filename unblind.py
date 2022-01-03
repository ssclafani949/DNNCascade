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
import utils


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
        assert len(trial) == 3

        # compute pre-trial p-value for given source
        bg = bgs[i]
        pval = bg.sf(trial[0], fit=False)
        pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
        trial.append(pval)
        trial.append(pval_nsigma)

        # print result
        msg = '#{:03d} | ts: {:5.2f} | ns: {:6.2f} | gamma: {:4.2f}'
        msg += ' | n-sigma: {:5.2f} | n-trials: {:7d} | {}'
        msg = msg.format(i, *trial[:3], pval_nsigma, len(bg), source[0])

        if pval_nsigma < 3.:
            print(msg)
        elif pval_nsigma < 5.:
            print(utils.bcolors.YELLOW + msg + utils.bcolors.ENDC)
        else:
            print(utils.bcolors.GREEN + msg + utils.bcolors.ENDC)

        # append results of this source to overall trials
        trials.append(trial)

    # get hottest source
    min_idx = np.argmin(np.array(trials)[:, 3])
    min_source = sourcelist[min_idx][0]
    pval_min = trials[min_idx][3]
    pval_min_nsigma = trials[min_idx][4]

    # compute trial-corrected p-value
    ts_mlog10p = -np.log10(pval_min)
    pval = bg_corr.sf(ts_mlog10p, fit=False)
    pval_nsigma = bg_corr.sf_nsigma(ts_mlog10p, fit=False)

    # print results to console
    utils.print_result(
        title='Results for source list',
        n_trials=len(bg_corr),
        trial=trials[min_idx],
        pval=pval,
        pval_nsigma=pval_nsigma,
        add_items={
            'hottest source': min_source,
            'pre-trial p-value': '{:3.3e}'.format(pval_min),
            'pre-trial n-sigma': '{:3.2f}'.format(pval_min_nsigma),
            'trial-factor': '{:3.2f}'.format(pval/pval_min),
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
@click.argument('template')
@click.option('--seed', default=None, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--TRUTH', default=None, type=bool,
              help='Must be Set to TRUE to unblind')
@pass_state
def unblind_gp(state, template, seed, cpus, truth, logging=True):
    """Unblind galactic plane templates kra5, kra50, pi0
    """

    # check if valid template name was passed
    template = template.lower()
    if template not in ['kra5', 'kra50', 'pi0']:
        if template == 'fermibubbles':
            msg = 'To unblind Fermi bubbles use: `unblind-fermibubbles`'
            raise ValueError(msg)
        else:
            raise ValueError('Unknown template: {}!'.format(template))

    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    ana = state.ana

    def get_tr(template_str, TRUTH):
        gp_conf = cg.get_gp_conf(
            template_str=template_str, base_dir=state.base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr = get_tr(template, TRUTH=truth)
    t0 = now()
    if truth:
        print('UNBLINDING!!!')

    # get background trials
    print('Loading BKG TRIALS')
    base_dir = state.base_dir + '/gp/trials/{}/{}/'.format(
        state.ana_name, template)
    sigfile = '{}/trials.dict'.format(base_dir)
    sig = np.load(sigfile, allow_pickle=True)
    bg = cy.dists.TSD(sig['poisson']['nsig'][0.0]['ts'])

    trial = tr.get_one_fit(TRUTH=truth,  seed=seed, logging=logging)
    pval = bg.sf(trial[0], fit=False)
    pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
    trial.append(pval)
    trial.append(pval_nsigma)

    # print results to console
    utils.print_result(
        title='Results for GP template: {}'.format(template),
        n_trials=len(bg), trial=trial, pval=pval, pval_nsigma=pval_nsigma,
    )
    flush()

    if truth:
        out_dir = cy.utils.ensure_dir('{}/gp/results/{}'.format(
            state.base_dir, template))
        out_file = '{}/{}_unblinded.npy'.format(
            out_dir, template)
        print ('-> {}'.format(out_file))
        np.save(out_file, trials)


@cli.command()
@click.option('--seed', default=None, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--TRUTH', default=None, type=bool,
              help='Must be Set to TRUE to unblind')
@pass_state
def unblind_fermibubbles(state, seed, cpus, truth, logging=True):
    """Unblind Fermi bubble templates with cutoffs 50/100/500 TeV
    """
    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    ana = state.ana

    cutoffs = [50, 100, 500]

    def get_tr(cutoff, TRUTH):
        cutoff_GeV = cutoff * 1e3
        gp_conf = cg.get_gp_conf(
            template_str='fermibubbles',
            cutoff_GeV=cutoff_GeV,
            base_dir=state.base_dir
        )
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    t0 = now()
    if truth:
        print('UNBLINDING!!!')

    # ---------------------
    # Get background trials
    # ---------------------
    # load correlated MultiTrialRunner background trials
    print('Loading correlated background trials...')
    base_dir = state.base_dir + '/gp/trials/{}/fermibubbles/'.format(
        state.ana_name)
    bgfile_corr = '{}/correlated_trials/correlated_bg.npy'.format(base_dir)
    trials_corr = np.load(bgfile_corr, allow_pickle=True)
    bg_corr = cy.dists.TSD(trials_corr)

    # load bkg trials for each cutoff
    print('Loading BKG TRIALS')
    sigfile = '{}/trials.dict'.format(base_dir)
    sig = np.load(sigfile, allow_pickle=True)
    bgs = []
    for cutoff in cutoffs:
        # safety check to make sure the correct bg trials exist
        # ToDo: csky.bk.get_best should be modified to check for maximum
        # allowed difference, if provided.
        c_keys = list(sig['poisson']['cutoff'].keys())
        if not np.any(np.abs([c - cutoff for c in c_keys]) < 1e-3):
            msg = 'Cutoff {:3.4f} does not exist in bg trials: {}!'
            raise ValueError(msg.format(cutoff, c_keys))

        # get background trials
        sig_trials = cy.bk.get_best(sig, 'poisson', 'cutoff', cutoff, 'nsig')
        bgs.append(cy.dists.TSD(sig_trials[0.0]['ts']))

    # -------------------
    # Unblind each cutoff
    # -------------------
    trials = []
    result_msgs = []
    for i, cutoff in enumerate(cutoffs):
        tr = get_tr(cutoff=cutoff, TRUTH=truth)

        trial = tr.get_one_fit(TRUTH=truth,  seed=seed, logging=logging)
        assert len(trial) == 2
        pval = bgs[i].sf(trial[0], fit=False)
        pval_nsigma = bgs[i].sf_nsigma(trial[0], fit=False)
        trial.append(pval)
        trial.append(pval_nsigma)

        # print result
        msg = (
            '    ts: {:5.2f} | ns: {:6.2f} | n-sigma: {:5.2f} '
            '| cutoff: {:5.1f} TeV | n-trials: {:7d}'
        ).format(*trial[:2], pval_nsigma, cutoff, len(bgs[i]))

        if pval_nsigma < 3.:
            result_msgs.append(msg)
        elif pval_nsigma < 5.:
            result_msgs.append(utils.bcolors.YELLOW + msg + utils.bcolors.ENDC)
        else:
            result_msgs.append(utils.bcolors.GREEN + msg + utils.bcolors.ENDC)

        # append results of this source to overall trials
        trials.append(trial)

    print()
    print('Pre-trial results for individual cutoffs:')
    for msg in result_msgs:
        print(msg)
    print()

    # --------------------
    # Trial-correct cutoff
    # --------------------
    # get most significant cutoff
    min_idx = np.argmin(np.array(trials)[:, 2])
    min_cutoff = cutoffs[min_idx]
    pval_min = trials[min_idx][2]
    pval_min_nsigma = trials[min_idx][3]

    # compute trial-corrected p-value
    ts_mlog10p = -np.log10(pval_min)
    pval = bg_corr.sf(ts_mlog10p, fit=False)
    pval_nsigma = bg_corr.sf_nsigma(ts_mlog10p, fit=False)

    # print results to console
    utils.print_result(
        title='Results for Fermi Bubble Template',
        n_trials=len(bg_corr),
        trial=trials[min_idx],
        pval=pval,
        pval_nsigma=pval_nsigma,
        add_items={
            'most significant cutoff': '{:3.1f} TeV'.format(min_cutoff),
            'pre-trial p-value': '{:3.3e}'.format(pval_min),
            'pre-trial n-sigma': '{:3.2f}'.format(pval_min_nsigma),
            'trial-factor': '{:3.2f}'.format(pval/pval_min),
        },
    )
    flush()

    if truth:
        out_dir = cy.utils.ensure_dir('{}/gp/results/fermibubbles'.format(
            state.base_dir))
        out_file = '{}/fermibubbles_unblinded.npy'.format(out_dir)
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
        utils.print_result(
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

    # load uncorrelated bg trials at each declination
    print('Loading bg trials at each declination...')
    base_dir = state.base_dir + '/ps/trials/DNNC'
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
        bgs = np.load(bgfile, allow_pickle=True)['dec']
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
        bg_trials = np.load(bgfile, allow_pickle=True)['dec']
        bgs = {key: cy.dists.TSD(trials) for key, trials in bg_trials.items()}

    # load correlated sky scan trials for trial-correction
    print('Loading correlated sky scan trials...')
    trials_corr_file = state.base_dir + '/skyscan/trials/DNNC/sky_scan_bg.npy'
    trials_corr_dict = np.load(trials_corr_file, allow_pickle=True)
    fit_str = 'fit' if fit else 'nofit'
    trials_corr = trials_corr_dict['nside'][nside]['bg'][fit_str]
    bg_corr_north = cy.dists.TSD(trials_corr.mlog10p_north)
    bg_corr_south = cy.dists.TSD(trials_corr.mlog10p_south)

    def ts_to_p(dec, ts):
        return cy.dists.ts_to_p(bgs, np.degrees(dec), ts, fit=fit)

    t0 = now()
    print('Running sky scan...')
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
    ss_trial = sstr.get_one_scan(logging=True, seed=seed, TRUTH=truth)

    # ------------------------
    # compute trial correction
    # ------------------------
    # ss_trial shape: [4, npix]
    # with [-log10(p), ts, ns, gamma] along first axis
    mlog10ps_sky = ss_trial[0]

    mask_north = utils.get_mask_north_dict([nside])[nside]

    # get hottest pixels
    mlog10ps_north = np.array(mlog10ps_sky)
    mlog10ps_north[~mask_north] = -np.inf
    ipix_max_north = np.argmax(mlog10ps_north)

    mlog10ps_south = np.array(mlog10ps_sky)
    mlog10ps_south[mask_north] = -np.inf
    ipix_max_south = np.argmax(mlog10ps_south)

    # compute trial-corrected p-values
    ts_mlog10ps_north = mlog10ps_sky[ipix_max_north]
    pval_north = bg_corr_north.sf(ts_mlog10ps_north, fit=False)
    pval_north_nsigma = bg_corr_north.sf_nsigma(ts_mlog10ps_north, fit=False)

    ts_mlog10ps_south = mlog10ps_sky[ipix_max_south]
    pval_south = bg_corr_south.sf(ts_mlog10ps_south, fit=False)
    pval_south_nsigma = bg_corr_south.sf_nsigma(ts_mlog10ps_south, fit=False)

    def ipix_to_dec_ra(ipix):
        theta, phi = hp.pixelfunc.pix2ang(nside, ipix)
        return np.pi/2. - theta, np.pi*2. - phi

    # print out results
    utils.print_result(
        title='Results for northern sky',
        n_trials=len(bg_corr_north),
        trial=ss_trial[:, ipix_max_north],
        pval=pval_north,
        pval_nsigma=pval_north_nsigma,
        add_items={
            'Gamma': ss_trial[3, ipix_max_north],
            'Location': 'Dec: {:3.2f}° | RA: {:3.2f}°'.format(
                *np.rad2deg(ipix_to_dec_ra(ipix_max_north))),
            'pre-trial p-value': '{:3.3e}'.format(10**(-ts_mlog10ps_north)),
            'trial-factor': '{:3.2f}'.format(
                pval_north/10**(-ts_mlog10ps_north)),
        },
    )

    utils.print_result(
        title='Results for southern sky',
        n_trials=len(bg_corr_south),
        trial=ss_trial[:, ipix_max_south],
        pval=pval_south,
        pval_nsigma=pval_south_nsigma,
        add_items={
            'Gamma': ss_trial[3, ipix_max_south],
            'Location': 'Dec: {:3.2f}° | RA: {:3.2f}°'.format(
                *np.rad2deg(ipix_to_dec_ra(ipix_max_south))),
            'pre-trial p-value': '{:3.3e}'.format(10**(-ts_mlog10ps_south)),
            'trial-factor': '{:3.2f}'.format(
                pval_south/10**(-ts_mlog10ps_south)),
        },
    )

    if truth:
        out_dir = cy.utils.ensure_dir('{}/skyscan/results/'.format(
            state.base_dir))
        out_file = '{}/unblinded_skyscan.npy'.format(
            out_dir,  seed)
        print ('-> {}'.format(out_file))
        results = {
            'ss_trial': ss_trial,
            'ipix_max_north': ipix_max_north,
            'ipix_max_south': ipix_max_south,
            'pval_north': pval_north,
            'pval_south': pval_south,
            'pval_north_nsigma': pval_north_nsigma,
            'pval_south_nsigma': pval_south_nsigma,
        }
        np.save(out_file, results)

        # plot skymaps
        utils.plot_ss_trial(ss_trial, outdir=out_dir)


if __name__ == '__main__':
    exe_t0 = now()
    print ('start at {} .'.format(exe_t0))
    cli()
