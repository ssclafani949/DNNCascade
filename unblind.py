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

    def get_tr(dec, ra,  truth):
        src = cy.utils.sources(ra, dec, deg=False)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf=conf, TRUTH=truth)
        return tr, src

    for source in sourcelist:

        tr, src = get_tr(
            dec=np.radians(source[2]),
            ra=np.radians(source[1]),
            truth=truth)

        if truth:
            print('UNBLINDING!!!')

        trial = tr.get_one_fit(TRUTH=truth, seed=seed, logging=logging)
        print(trial)
        print(source[0], trial[0])
    t1 = now()
    flush()
    trials.append(trial)
    if truth:
        out_dir = cy.utils.ensure_dir('{}/ps/results/'.format(
            state.base_dir, state.ana_name))
        out_file = '{}/fulllist_unblinded.npy'.format(
            out_dir)
        print ('-> {}'.format(out_file))
        np.save(out_file, trials)


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
    print('Number of Background Trials: {}'.format(len(bg)))
    trial = tr.get_one_fit(TRUTH=truth,  seed=seed, logging=logging)
    print('TS: {} ns: {}'.format(trial[0], trial[1]))
    pval = bg.sf(trial[0], fit=False)
    pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
    print('pval : {}'.format(pval))
    trial.append(pval)
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
        print('Number of Background Trials: {}'.format(len(bg)))
        trial = tr.get_one_fit(TRUTH=truth,  seed=seed, logging=logging)
        print('TS: {} ns: {}'.format(trial[0], trial[1]))
        pval = bg.sf(trial[0], fit=False)
        pval_nsigma = bg.sf_nsigma(trial[0], fit=False)
        print('pval : {}'.format(pval))
        trial.append(pval)
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
