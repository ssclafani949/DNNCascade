#!/usr/bin/env python

import csky as cy
import numpy as np
import healpy as hp
import pickle, datetime, socket
import histlite as hl
import astropy
now = datetime.datetime.now
import matplotlib.pyplot as plt
import click, sys, os, time
import config as cg
flush = sys.stdout.flush
hp.disable_warnings()

repo, ana_dir, base_dir, job_basedir = cg.repo, cg.ana_dir, cg.base_dir, cg.job_basedir

class State (object):
    def __init__ (self, ana_name, ana_dir, save, base_dir, job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = ana_name, ana_dir, save, job_basedir
        self.base_dir = base_dir
        self._ana = None

    @property
    def ana (self):
        if self._ana is None:
            repo.clear_cache()
            if 'baseline' in base_dir:
                specs = cy.selections.DNNCascadeDataSpecs.DNNC_10yr
            elif 'systematics' in base_dir:
                specs = cy.selections.DNNCascadeDataSpecs.DNNC_10yr_systematics
            ana = cy.get_analysis(repo, 'version-001-p00', specs, dir = base_dir)
            if self.save:
                cy.utils.ensure_dir (self.ana_dir)
                ana.save (self.ana_dir)
            ana.name = self.ana_name
            self._ana = ana
        return self._ana

    @property
    def state_args (self):
        return '--ana {} --ana-dir {} --base-dir {}'.format (
            self.ana_name, self.ana_dir, self.base_dir)

pass_state = click.make_pass_decorator (State)

@click.group (invoke_without_command=True, chain=True)
@click.option ('-a', '--ana', 'ana_name', default='DNNC', help='Dataset title')
@click.option ('--ana-dir', default=ana_dir, type=click.Path ())
@click.option ('--job_basedir', default=job_basedir, type=click.Path ())
@click.option ('--save/--nosave', default=False)
@click.option ('--base-dir', default=base_dir,
               type=click.Path (file_okay=False, writable=True))
@click.pass_context
def cli (ctx, ana_name, ana_dir, save, base_dir, job_basedir):
    ctx.obj = State.state = State (ana_name, ana_dir, save, base_dir, job_basedir)


@cli.resultcallback ()
def report_timing (result, **kw):
    exe_t1 = now ()
    print ('c11: end at {} .'.format (exe_t1))
    print ('c11: {} elapsed.'.format (exe_t1 - exe_t0))

@cli.command ()
@pass_state
def setup_ana (state):
    state.ana

@cli.command()
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@click.option ('--TRUTH', default=None, type=bool, help='Must be Set to TRUE to unblind')
@pass_state
def unblind_sourcelist ( 
        state, seed, truth,  logging=True):
    """
    Unblind Source List
    """
    sourcelist = np.load('catalogs/Source_List_DNNC.npy', allow_pickle = True) 
    t0 = now ()
    print(truth)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir, dec_deg))

    def get_tr(dec, ra, cpus, truth):
        src = cy.utils.sources(ra, dec, deg=False)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus, TRUTH=TRUTH)
        return tr, src

    for source in sourcelist:
        print('Source {}'.format(src[0]))
        
        tr , src = get_tr(
            dec = np.radians(source[2]), 
            ra=np.radians(source[1]), 
            cpus=cpus, TRUTH=TRUTH)
        if TRUTH:
            print('UNBLINDING!!!')
        
        trial =  tr.get_one_fit (TRUTH=truth,  seed = seed, logging=logging)
        print(trial )
        print(source[0], trial[0])
    t1 = now ()
    flush ()
    out_dir = cy.utils.ensure_dir ('{}/ps/results/'.format (
        state.base_dir, state.ana_name))
    out_file = '{}/fulllist_unblinded.npy'.format (
        out_dir)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)


@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_gp_trials ( 
            state, temp, n_trials, n_sig, 
            poisson, seed, cpus, 
            cutoff, logging=True):
    """
    Do trials for galactic plane templates including fermi bubbles
    and save output in a structured dirctory based on paramaters
    """
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    ana = state.ana
    cutoff_GeV = cutoff * 1e3

    def get_tr(temp):
        gp_conf = cg.get_gp_conf(
            temp=temp,
            cutoff_GeV=cutoff_GeV,
            base_dir=state.base_dir,
        )
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr = get_tr(temp)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()
    trials = tr.get_many_fits (
        n_trials, n_sig=n_sig, poisson=poisson, seed=seed, logging=logging)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))
    print (trials if n_sig else cy.dists.Chi2TSD (trials))
    print (t1 - t0, 'elapsed.')
    flush ()
    if temp =='fermibubbles':
        out_dir = cy.utils.ensure_dir (
            '{}/gp/trials/{}/{}/{}/cutoff/{}/nsig/{:08.3f}'.format (
                state.base_dir, state.ana_name,
                temp,
                'poisson' if poisson else 'nonpoisson', cutoff,
                n_sig))
    else:
        out_dir = cy.utils.ensure_dir (
            '{}/gp/trials/{}/{}/{}/nsig/{:08.3f}'.format (
                state.base_dir, state.ana_name,
                temp,
                'poisson' if poisson else 'nonpoisson',
                n_sig))

    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)


@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--catalog',   default='snr' , type=str, help='Stacking Catalog, SNR, PWN or UNID')
@click.option ('--gamma', default=2.0, type=float, help = 'Spectrum to Inject')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def do_stacking_trials (
        state, n_trials, gamma, cutoff, catalog,
        n_sig,  poisson, seed, cpus, logging=True):
    """
    Do trials from a stacking catalog
    """
    print('Catalog: {}'.format(catalog))
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    cat = np.load('catalogs/{}_ESTES_12.pickle'.format(catalog),
        allow_pickle=True)
    src = cy.utils.Sources(dec=cat['dec_deg'], ra=cat['ra_deg'], deg=True)
    cutoff_GeV = cutoff * 1e3
    dir = cy.utils.ensure_dir ('{}/{}/'.format (state.base_dir, catalog))
    def get_tr(src, gamma, cpus):
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr
    tr = get_tr(src, gamma, cpus)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()
    trials = tr.get_many_fits (
        n_trials, n_sig=n_sig, poisson=poisson, seed=seed, logging=logging)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))
    print (trials if n_sig else cy.dists.Chi2TSD (trials))
    print (t1 - t0, 'elapsed.')
    flush ()
    if n_sig:
        out_dir = cy.utils.ensure_dir (
            '{}/stacking/trials/{}/catalog/{}/{}/gamma/{:.3f}/cutoff_TeV/{:.0f}/nsig/{:08.3f}'.format (
                state.base_dir, state.ana_name, catalog,
                'poisson' if poisson else 'nonpoisson',
                 gamma, cutoff,  n_sig))
    else:
        out_dir = cy.utils.ensure_dir ('{}/stacking/trials/{}/catalog/{}/bg/'.format (
            state.base_dir, state.ana_name, catalog))
    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)



if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
