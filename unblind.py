#!/usr/bin/env python

import csky as cy
import numpy as np
import healpy as hp
import pickle, datetime, socket
import histlite as hl
import astropy
now = datetime.datetime.now
import matplotlib.pyplot as plt
import pandas as pd
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
    trials = []
    sourcelist = np.load('catalogs/Source_List_DNNC.npy', allow_pickle = True) 
    t0 = now ()
    print(truth)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir))
    def get_tr(dec, ra,  truth):
        src = cy.utils.sources(ra, dec, deg=False)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf= conf, TRUTH=truth)
        return tr, src

    for source in sourcelist:
        
        tr , src = get_tr(
            dec = np.radians(source[2]), 
            ra=np.radians(source[1]), 
            truth = truth)
        
        if truth:
            print('UNBLINDING!!!')
        
        trial =  tr.get_one_fit (TRUTH=truth,  seed = seed, logging=logging)
        print(trial )
        print(source[0], trial[0])
    t1 = now ()
    flush ()
    trials.append(trial)
    if truth:
        out_dir = cy.utils.ensure_dir ('{}/ps/results/'.format (
            state.base_dir, state.ana_name))
        out_file = '{}/fulllist_unblinded.npy'.format (
            out_dir)
        print ('-> {}'.format (out_file))
        np.save (out_file, trials)


@cli.command()                                                                              
@click.option ('--sourcenum', default=1, type=int, help='what source in the list')
@click.option ('--ntrials', default=1000, type=int, help='number of trials to run')
@click.option ('--cpus', default=1, type=int, help='ncpus')
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_bkg_trials_sourcelist ( 
        state, sourcenum, ntrials, cpus, seed, logging=True):
    """ 
    Do Background Trials at the exact declination of each source. 
    Used as an input for the MTR correlated trials to correctly calculate 
    pre-trial pvalues
    """
    sourcelist = pd.read_pickle('catalogs/Source_List_DNNC.pickle')
    ras = sourcelist.RA.values
    decs = sourcelist.DEC.values
    names = sourcelist.Names.values
   
    if seed is None:
        seed = int (time.time () % 2**32)
 
    t0 = now ()
    ana = state.ana
    def get_tr(dec, ra, cpus=cpus):
        src = cy.utils.sources(ra=ra, dec=dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus )
        return tr                                                        
    dec = decs[sourcenum]
    ra = ras[sourcenum]
    name = names[sourcenum] 
    
    tr = get_tr(dec=dec, ra=ra, cpus=cpus)

    print('Doing Background Trials for Source {} : {}'.format(sourcenum, name))
    print('DEC {} : RA {}'.format(dec, ra))
    print('Seed: {}'.format(seed))
    trials = tr.get_many_fits(ntrials, seed=seed)
    t1 = now ()
    flush ()
    out_dir = cy.utils.ensure_dir ('{}/ps/correlated_trials/bg/source_{}'.format (
        state.base_dir, sourcenum))
    out_file = '{}/bkg_trials_{}_seed_{}.npy'.format (
        out_dir, ntrials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)
   
@cli.command()
@pass_state
def collect_bkg_trials_sourcelist( state ):
    """
    Collect all the background trials from do-bkg-trials-sourcelist into one list to feed into 
    the MTR for calculation of pre-trial pvalues.  Unlike do-ps-trials, these are done at the exact
    declination of each source
    """
    bgs = []
    numsources = 109
    base_dir = '{}/ps/correlated_trials/bg/'.format(state.base_dir)
    for i in range(numsources):
        bg = cy.bk.get_all(base_dir +'source_{}/'.format(i), '*.npy',
            merge=np.concatenate, post_convert =  (lambda x: cy.dists.TSD(cy.utils.Arrays(x))))
        bgs.append(bg)
    np.save('{}/ps/correlated_trials/pretrial_bgs.npy'.format(state.base_dir), bgs)

                                                                       
@cli.command()
@click.option ('--n-trials', default=10, type=int, help='number of trials to run')
@click.option ('--cpus', default=1, type=int, help='ncpus')
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_correlated_trials_sourcelist ( 
        state, n_trials, cpus, seed,  logging=True):
    """
    Use MTR for correlated background trials evaluating at each source in the sourcelist
    """
    sourcelist = pd.read_pickle('catalogs/Source_List_DNNC.pickle') 
    ras = sourcelist.RA.values
    decs = sourcelist.DEC.values
    if seed is None:
        seed = int (time.time () % 2**32)

    t0 = now ()
    ana = state.ana
    print('Loading Backgrounds')
    bgs = np.load('{}/ps/correlated_trials/pretrial_bgs.npy'.format(state.base_dir), allow_pickle=True)
    def get_tr(dec, ra, cpus=cpus):
        src = cy.utils.sources(ra=ra, dec=dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr
    print('Getting trial runners')
    trs = [get_tr(d,r) for d,r in zip(decs, ras)]
    tr_inj = trs[0] 
    multr = cy.trial.MultiTrialRunner(
        ana,
        # bg+sig injection trial runner (produces trials)
        tr_inj,
        # llh test trial runners (perform fits given trials)
        trs,
        # background distrubutions
        bgs=bgs,
        # use multiprocessing
        mp_cpus=cpus,
    ) 
    trials = multr.get_many_fits(n_trials)
    t = trials.as_dataframe
    t1 = now ()
    flush ()
    out_dir = cy.utils.ensure_dir ('{}/ps/correlated_trials/correlated_bg/'.format (
        state.base_dir, state.ana_name))
    out_file = '{}/correlated_trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    t.to_pickle (out_file)

@cli.command()
@click.argument('temp')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--TRUTH', default=None, type=bool, help='Must be Set to TRUE to unblind')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def unblind_gp ( 
            state, temp,  
            seed, cpus, 
            truth, cutoff, logging=True):
    """
    Unblind a particular galactic plane templaet
    """
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    ana = state.ana
    cutoff_GeV = cutoff * 1e3

    def get_tr(template_str, TRUTH):
        gp_conf = cg.get_gp_conf(
            template_str=template_str,
            cutoff_GeV=cutoff_GeV,
            base_dir=state.base_dir
        )
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr = get_tr(temp, TRUTH=truth)
    t0 = now ()
    if truth:
        print('UNBLINDING!!!')
   
    print('Loading BKG TRIALS')
    base_dir = state.base_dir + '/gp/trials/{}/{}/'.format(state.ana_name, temp)
    sigfile = '{}/trials.dict'.format (base_dir)                                       
    sig = np.load (sigfile, allow_pickle=True)
    if temp == 'fermibubbles':
        sig_trials = cy.bk.get_best(sig, 'poisson', 'cutoff', cutoff, 'nsig')
    else:
        sig_trials = cy.bk.get_best(sig, 'poisson', 'nsig')
    b = sig_trials[0.0]['ts']
    bg = cy.dists.TSD(b) 
    print('Number of Background Trials: {}'.format(len(bg)))
    trial =  tr.get_one_fit (TRUTH=truth,  seed = seed, logging=logging)
    print('TS: {} ns: {}'.format(trial[0], trial[1]))
    pval = np.mean(bg.values > trial[0])
    print('pval : {}'.format(pval))
    trial.append(pval)
    flush ()
    if truth:
        if temp == 'fermibubbles':
            out_dir = cy.utils.ensure_dir ('{}/gp/results/{}/'.format (
                state.base_dir, temp))
            out_file = '{}/{}_cutoff_{}_unblinded.npy'.format (
                out_dir, temp, cutoff)
        else:
            out_dir = cy.utils.ensure_dir ('{}/gp/results/{}/'.format (
                state.base_dir, temp))
            out_file = '{}/{}_unblinded.npy'.format (
                out_dir, temp)
        print ('-> {}'.format (out_file))
        np.save (out_file, trials)


@cli.command()
@click.option ('--TRUTH', default=None, type=bool, help='Must be Set to TRUE to unblind')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--seed', default=None, type=int)
@pass_state
def unblind_stacking (
        state, 
        truth, cutoff, seed, logging=True):
    """
    Unblind all the stacking catalogs
    """
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    def get_tr(src, TRUTH):
        conf = cg.get_ps_conf(src=src, gamma=2.0, cutoff_GeV=np.inf)
        tr = cy.get_trial_runner(ana=ana, conf= conf, TRUTH=TRUTH)
        return tr
    if truth:
        print('UNBLINDING!!!')
    for catalog in ['snr', 'pwn', 'unid']:
        print('Catalog: {}'.format(catalog))
        cat = np.load('catalogs/{}_ESTES_12.pickle'.format(catalog),
            allow_pickle=True)
        src = cy.utils.Sources(dec=cat['dec_deg'], ra=cat['ra_deg'], deg=True)
        tr = get_tr(src, TRUTH=truth)
        
        print('Loading BKG TRIALS')
        base_dir = state.base_dir + '/stacking/'
        bgfile = '{}/{}_bg.dict'.format(base_dir, catalog)
        b = np.load (bgfile, allow_pickle=True)
        bg = cy.dists.TSD(b) 
        print('Number of Background Trials: {}'.format(len(bg)))
        trial =  tr.get_one_fit (TRUTH=truth,  seed = seed, logging=logging)
        print('TS: {} ns: {}'.format(trial[0], trial[1]))
        pval = np.mean(bg.values > trial[0])
        print('pval : {}'.format(pval))
        trial.append(pval)
        flush ()
        if truth:
            out_dir = cy.utils.ensure_dir ('{}/stacking/results/{}/'.format (
                state.base_dir, catalog))
            out_file = '{}/{}_unblinded.npy'.format (
                out_dir, catalog)
            print ('-> {}'.format (out_file))
            np.save (out_file, trials)


@cli.command ()
@click.option('--nside', default=128, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--seed', default=None, type = int)
@click.option ('--TRUTH', default=None, type=bool, help='Must be Set to TRUE to unblind')
@pass_state
def unblind_skyscan(state, 
                    nside, cpus, seed, truth):
    """
    Unblind the skyscan and save the true map
    """

    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    base_dir = state.base_dir + '/ps/trials/DNNC'
    bgfile = '{}/bg_chi2.dict'.format (base_dir)
    bg = np.load (bgfile, allow_pickle=True)
    ts_to_p = lambda dec, ts: cy.dists.ts_to_p (bg['dec'], np.degrees(dec), ts, fit=True)
    t0 = now ()
    ana = state.ana
    conf = cg.get_ps_conf(src=None, gamma=2.0)
    conf.pop('src')
    conf.update({
        'ana': ana,
        'mp_cpus': cpus,
        'extra_keep': ['energy'],
    })                                                                                                           
    sstr = cy.get_sky_scan_trial_runner(conf=conf, TRUTH=truth, 
                                        min_dec= np.radians(-80),
                                        max_dec = np.radians(80),
                                        mp_scan_cpus = cpus,
                                        nside=nside, ts_to_p = ts_to_p)        
    if truth:
        print('UNBLINDING!!!!')
    trials = sstr.get_one_scan(logging=True, seed=seed, TRUTH=truth)
    if truth:
        out_dir = cy.utils.ensure_dir ('{}/skyscan/results/'.format (      
            state.base_dir))
        out_file = '{}/unblinded_skyscan.npy'.format (
            out_dir,  seed)
        print ('-> {}'.format (out_file))
        np.save (out_file, trials)




if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
