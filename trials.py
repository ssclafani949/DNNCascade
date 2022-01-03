#!/usr/bin/env python

import csky as cy
import numpy as np
import pandas as pd
import glob
import healpy as hp
import pickle, datetime, socket
import histlite as hl
now = datetime.datetime.now
import matplotlib.pyplot as plt
import click, sys, os, time
import config as cg
import utils
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
                specs = cy.selections.DNNCascadeDataSpecs.DNNC_10yr_systematics_full
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
@click.option('--n-trials', default=1000, type=int, help='Number of bg trials to run')
@click.option ('--gamma', default=2.0, type=float, help='Spectral Index to inject')
@click.option ('--sigsub/--nosigsub', default=True, type=bool, 
    help='Include Signal Subtraction in LLH')
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--seed', default=None, type=int, help='Seed for scrambeling')
@click.option ('--cpus', default=1, type=int, help='Number of CPUs to use')
@click.option ('--nsigma', default=None, type=float, help='Do DP trials')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_ps_sens ( 
        state, n_trials, gamma, sigsub, dec_deg, seed, 
        cpus, nsigma, cutoff, logging=True):
    """
    Do seeded point source sensitivity and save output.  Useful for quick debugging not for 
    large scale trial calculations.
    """
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    ana = state.ana
    sindec = np.sin(np.radians(dec_deg))
    def get_PS_sens(sindec, n_trials=n_trials, gamma=gamma, mp_cpu=cpus):

        def get_tr(sindec, gamma, cpus):
            src = cy.utils.sources(0, np.arcsin(sindec), deg=False)
            cutoff_GeV = cutoff * 1e3
            conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)

            # overwrite sigsub setting
            conf['sigsub'] = sigsub

            tr = cy.get_trial_runner(ana=ana, conf=conf, mp_cpus=cpus)
            return tr, src

        tr, src = get_tr(sindec, gamma, cpus)
        print('Performing BG Trails at RA: {}, DEC: {}'.format(src.ra_deg, src.dec_deg))
        bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials, mp_cpus=cpus, seed=seed))
        if nsigma:
            beta = 0.5
            ts = bg.isf_nsigma(nsigma)
            n_sig_step = 25
        else:
            beta = 0.9
            ts = bg.median()
            n_sig_step = 6
        sens = tr.find_n_sig(
            # ts, threshold
            ts,
            # beta, fraction of trials which should exceed the threshold
            beta,
            # n_inj step size for initial scan
            n_sig_step=n_sig_step,
            # this many trials at a time
            batch_size=2500,
            # tolerance, as estimated relative error
            tol=.025,
            first_batch_size = 250,
            mp_cpus=cpus,
            seed=seed
        )
        sens['flux'] = tr.to_E2dNdE (sens['n_sig'], E0=100, unit=1e3)
        print(sens['flux'])
        return sens

    t0 = now ()
    print ('Beginning calculation at {} ...'.format (t0))
    flush ()
    sens = get_PS_sens (sindec, gamma=gamma, n_trials=n_trials) 
    
    sens_flux = np.array(sens['flux'])
    out_dir = cy.utils.ensure_dir('{}/E{}/{}/dec/{:+08.3f}/'.format(
        state.base_dir, int(gamma*100), 'sigsub' if sigsub else 'nosigsub',  dec_deg))
    if nsigma:
        out_file = out_dir + 'dp_{}sigma.npy'.format(nsigma)
    else:
        out_file = out_dir + 'sens.npy'
    print(sens_flux)
    np.save(out_file, sens_flux)
    t1 = now ()
    print ('Finished sens at {} ...'.format (t1))

@cli.command()
@click.option('--n-trials', default=1000, type=int, help='Number of trails to run')
@click.option ('-n', '--n-sig', default=0, type=float, help = 'Number of signal events to inject')
@click.option ('--poisson/--nopoisson', default=True, 
    help = 'toggle possion weighted signal injection')
@click.option ('--sigsub/--nosigsub', default=True, type=bool, 
    help='Include Signal Subtraction in LLH')
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--gamma', default=2.0, type=float, help='Spectral Index to inject')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@click.option ('--cpus', default=1, type=int, help='Number of CPUs to use')
@pass_state
def do_ps_trials ( 
        state, dec_deg, n_trials, gamma, cutoff, n_sig, 
        poisson, sigsub, seed, cpus, logging=True):
    """
    Do seeded point source trials and save output in a structured dirctory based on paramaters
    Used for final Large Scale Trail Calculation
    """
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    dec = np.radians(dec_deg)
    replace = False 
    sindec = np.sin(dec)
    t0 = now ()
    ana = state.ana
    src = cy.utils.Sources(dec=dec, ra=0)
    cutoff_GeV = cutoff * 1e3
    dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir, dec_deg))
    a = ana[0]

    def get_tr(sindec, gamma, cpus):
        src = cy.utils.sources(0, np.arcsin(sindec), deg=False)
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr, src

    tr , src = get_tr(sindec, gamma, cpus)
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
            '{}/ps/trials/{}/{}/{}/gamma/{:.3f}/cutoff_TeV/{:.0f}/dec/{:+08.3f}/nsig/{:08.3f}'.format (
                state.base_dir, state.ana_name,
                'sigsub' if sigsub else 'nosigsub',
                'poisson' if poisson else 'nonpoisson',
                 gamma, cutoff, dec_deg, n_sig))
    else:
        out_dir = cy.utils.ensure_dir ('{}/ps/trials/{}/bg/dec/{:+08.3f}/'.format (
            state.base_dir, state.ana_name, dec_deg))
    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)


@cli.command ()
@click.option ('--fit/--nofit', default=True, help = 'Chi2 Fit or Not')
@click.option ('--dist/--nodist', default=True, help = 'Distribution is TSD or leave in arrays')
@click.option ('--inputdir', default=None, help = 'Option to set a read directory that isnt the base directory')
@pass_state
def collect_ps_bg (state, fit,  dist, inputdir):
    """
    Collect all Background Trials and save in nested dict
    """
    kw = {}
    dec_degs = np.r_[-81:+81.01:2]
    if fit:
        TSD = cy.dists.Chi2TSD
        suffix = '_chi2'
    else:
        if dist:
            TSD = cy.dists.TSD
            suffix = 'TSD'
        else:
            suffix = ''
    outfile = '{}/ps/trials/{}/bg{}.dict'.format (
        state.base_dir, state.ana_name,  suffix)
    bg = {}
    bgs = {}
    if inputdir:
        bg_dir = inputdir
    else: 
        bg_dir = '{}/ps/trials/{}/bg'.format (
            state.base_dir, state.ana_name)
    for dec_deg in dec_degs:
        key = '{:+08.3f}'.format (dec_deg)
        flush ()
        print('{}/dec/{}/'.format(bg_dir, key))
        if dist == False:
            print('no dist') 
            post_convert = (lambda x: cy.utils.Arrays (x))
        else:
            post_convert = (lambda x: TSD (cy.utils.Arrays (x), **kw))
        bg_trials = cy.bk.get_all (
                '{}/dec/{}/'.format (bg_dir, key), '*.npy',
                merge=np.concatenate, post_convert=post_convert)
        if bg_trials is not False:
            bgs[float(key)] = bg_trials
    bg['dec'] = bgs
    print ('\rDone.' + 20 * ' ')
    flush ()
    print ('->', outfile)
    with open (outfile, 'wb') as f:
        pickle.dump (bg, f, -1)

@cli.command ()
@click.option ('--inputdir', default=None, help = 'Option to set a read directory that isnt the base directory')
@pass_state
def collect_ps_sig (state, inputdir):
    """
    Collect all Signal Trials and save in nested dict
    """
    if inputdir:
        sig_dir = inputdir
    else: 
        sig_dir = '{}/ps/trials/{}/sigsub/poisson'.format (state.base_dir, state.ana_name)
    sig = cy.bk.get_all (
        sig_dir, '*.npy', merge=np.concatenate, post_convert=cy.utils.Arrays)
    outfile = '{}/ps/trials/{}/sig.dict'.format (state.base_dir, state.ana_name)
    with open (outfile, 'wb') as f:
        pickle.dump (sig, f, -1)
    print ('->', outfile)


@cli.command ()
@click.option ('--gamma', default=2.0, type=float, help='Spectral Index to inject')
@click.option ('--nsigma', default=None, type=float, help='Number of sigma to find')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--verbose/--noverbose', default=False, help = 'Noisy Output')
@click.option ('--fit/--nofit', default=False, help = 'Fit the bkg dist to a chi2 or not?')
@click.option ('--inputdir', default=None, help = 'Option to set a read directory that isnt the base directory')
@pass_state
def find_ps_n_sig(state, nsigma, cutoff, gamma, verbose, fit, inputdir):
    """
    Calculate the Sensitvity or discovery potential once bg and sig files are collected
    """
    ana = state.ana
    if inputdir:
        indir = inputdir
    else: 
        indir = state.base_dir + '/ps/trials/DNNC'
    base_dir = state.base_dir + '/ps/trials/DNNC'
    sigfile = '{}/sig.dict'.format (indir)
    sig = np.load (sigfile, allow_pickle=True)
    bgfile = '{}/bg.dict'.format (indir)
    bg = np.load (bgfile, allow_pickle=True)
    decs = list(bg['dec'].keys())
    def get_n_sig(
                dec, gamma,
                beta=0.9, nsigma=None, cutoff=cutoff, fit=fit, verbose=verbose,
            ):
        if cutoff == None:
            cutoff_GeV = np.inf
            cutoff = np.inf
        else:
            cutoff_GeV = cutoff*1e3
        if verbose:
            print(gamma, dec, cutoff)
        sig_trials = cy.bk.get_best(sig,  'gamma', gamma, 'cutoff_TeV', 
            cutoff, 'dec', dec, 'nsig')    
        b = cy.bk.get_best(bg,  'dec', dec)
        if verbose:
            print(b)
        src = cy.utils.sources(0, dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf=conf)
            # determine ts threshold
        if nsigma !=None:
            #print('sigma = {}'.format(nsigma))
            if fit:
                ts = cy.dists.Chi2TSD(b).isf_nsigma(nsigma)
            else:
                ts = cy.dists.TSD(b).isf_nsigma(nsigma)
        else:
            #print('Getting sensitivity')
            ts = cy.dists.Chi2TSD(b).median()
        if verbose:
            print(ts)

        # include background trials in calculation
        trials = {0: b}
        trials.update(sig_trials)

        result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=verbose, trials=trials)
        flux = tr.to_E2dNdE(result['n_sig'], E0=100, unit=1e3)
        # return flux
        if verbose:
            print(ts, beta, result['n_sig'], flux)
        return flux , result['n_sig'], ts
    fluxs = []
    ns = []
    tss = []
    if fit:
        print('Fitting to a chi2')
        fit_str = 'chi2fit'
    else:
        print('Not fitting to a chi2 - using bkg trials')
        fit_str = 'nofit'    
    if nsigma:
        beta = 0.5
    else:
        beta = 0.9
    for i, dec in enumerate(decs):
        f, n, ts = get_n_sig(
            dec=dec, gamma=gamma, beta=beta, nsigma=nsigma, cutoff=cutoff,
            fit=fit, verbose=verbose,
        )
        print('{:.3} : {:.3} : {:.5}  : TS : {:.5}                                    '.format(
            dec, n, f, ts) , end='\r', flush=True)

        fluxs.append(f)
        ns.append(n)
        tss.append(ts)
    if nsigma:
        np.save(base_dir + '/ps_dp_{}sigma_flux_E{}_{}.npy'.format(
            nsigma, int(gamma * 100), fit_str), fluxs)
        np.save(base_dir + '/ps_dp_{}sigma_tss_E{}_{}.npy'.format(nsigma, int(gamma * 100), fit_str), tss)
        np.save(base_dir + '/ps_dp_{}sigma_nss_E{}_{}.npy'.format(nsigma, int(gamma * 100), fit_str), ns)
        np.save(base_dir + '/ps_dp_{}sigma_decs_E{}_{}.npy'.format(nsigma, int(gamma * 100), fit_str), decs)
    else:

        np.save(base_dir + '/ps_sens_flux_E{}_{}.npy'.format(int(gamma * 100), fit_str), fluxs)
        np.save(base_dir + '/ps_sens_nss_E{}_{}.npy'.format(int(gamma * 100), fit_str), ns)
        np.save(base_dir + '/ps_sens_decs_E{}_{}.npy'.format(int(gamma * 100), fit_str), decs)

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
    Do trials for galactic plane templates including Fermi bubbles
    and save output in a structured directory based on parameters
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print('Seed: {}'.format(seed))
    ana = state.ana
    cutoff_GeV = cutoff * 1e3

    def get_tr(temp):
        gp_conf = cg.get_gp_conf(
            template_str=temp, cutoff_GeV=cutoff_GeV, base_dir=state.base_dir)
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
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--gamma', default=2.5, type=float)
@click.option ('--nsigma', default=0, type=int)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_gp_sens ( 
        state, temp, n_trials,  seed, cpus, gamma, nsigma,
        cutoff, logging=True):
    """
    Calculate for galactic plane templates including fermi bubbles
    Recommend to use do_gp_trials for analysis level mass trial calculation
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
    cutoff_GeV = cutoff * 1e3

    def get_tr(temp):
        gp_conf = cg.get_gp_conf(
            template_str=temp,
            gamma=gamma,
            cutoff_GeV=cutoff_GeV,
            base_dir=state.base_dir,
        )
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr = get_tr(temp)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()

    bg = cy.dists.Chi2TSD(tr.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))
    if nsigma == 0:
        template_sens = tr.find_n_sig(
                        bg.median(), 
                        0.9, #percent above threshold (0.9 for sens)
                        n_sig_step=10,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    else:
        template_sens = tr.find_n_sig(
                        bg.isf_nsigma(nsigma),
                        0.5, #percent above threshold (0.5 for dp)
                        n_sig_step=50,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    
    if temp == 'pi0':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        template_sens['fluxE2_100TeV_GeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 1e5 , unit = 1)
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/gamma/{}'.format(
            state.base_dir,temp, gamma))
    elif temp == 'fermibubbles':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        template_sens['flux_1TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 1 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/gamma/{}/cutoff/{}_TeV/'.format(
            state.base_dir,temp, gamma, cutoff))
    else:
        template_sens['model_norm'] = tr.to_model_norm(template_sens['n_sig'])
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/'.format(
            state.base_dir,temp))

    flush ()
    print(cutoff_GeV) 
    if nsigma == 0:
        out_file = out_dir + 'sens.npy'
    else: 
        out_file = out_dir + 'dp_{}sigma.npy'.format(nsigma)

    print(template_sens)
    np.save(out_file, template_sens)
    print ('-> {}'.format (out_file))                                                          

@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--emin', default=500, type=float)
@click.option ('--emax', default=8.00, type=float)
@click.option ('--nsigma', default=0, type=int)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_gp_sens_erange ( 
        state, temp, n_trials,  seed, cpus, emin, emax, nsigma,
        cutoff, logging=True):
    """
    Same as do_gp_sens with an option to set the emin and emax, 
    Usefull if you want to calculate the relavant 90% enegy range by varying these paramaters
    """
    temp = temp.lower()
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    ana_lim = state.ana
    a = ana_lim[0]
    mask = (a.sig.true_energy > emin) & (a.sig.true_energy < emax)
    ana_lim[0].sig = a.sig[mask]
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
    cutoff_GeV = cutoff * 1e3

    def get_tr(temp, ana):
        gp_conf = cg.get_gp_conf(
            template_str=temp, cutoff_GeV=cutoff_GeV, base_dir=state.base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    tr_bg = get_tr(temp, ana)
    tr_lim = get_tr(temp, ana_lim)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()

    bg = cy.dists.Chi2TSD(tr_bg.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))
    if nsigma == 0:
        template_sens = tr_lim.find_n_sig(
                        bg.median(), 
                        0.9, #percent above threshold (0.9 for sens)
                        n_sig_step=10,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    else:
        template_sens = tr_lim.find_n_sig(
                        bg.isf_nsigma(nsigma),
                        0.5, #percent above threshold (0.9 for sens)
                        n_sig_step=15,
                        batch_size = n_trials / 3, 
                        tol = 0.02)
    
    if temp == 'pi0':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
    else:
        template_sens['model_norm'] = tr_bg.to_model_norm(template_sens['n_sig'])

        print(tr_bg.to_model_norm(template_sens['n_sig']))
        print(tr_lim.to_model_norm(template_sens['n_sig']))
    flush ()

    out_dir = cy.utils.ensure_dir(
        '{}/gp/{}/limited_Erange/'.format(
        state.base_dir,temp))
    if nsigma == 0:
        out_file = out_dir + 'Emin_{:.4}_Emax_{:.4}_sens.npy'.format(emin, emax)
    else: 
        out_file = out_dir + 'dp{}.npy'.format(nsigma)

    print(template_sens)
    np.save(out_file, template_sens)
    print ('-> {}'.format (out_file))                                                          

@cli.command ()
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def collect_gp_trials (state, inputdir):
    """
    Collect all Background and Signal Trials and save in nested dict
    """
    templates = ['fermibubbles', 'pi0', 'kra5', 'kra50']
    for template in templates:
        print(template)
        if inputdir:
            indir = inputdir
        else: 
            indir = '{}/gp/trials/{}/{}/'.format(state.base_dir, state.ana_name, template) 
        bg = cy.bk.get_all (
            indir,
            'trials*npy',
            merge=np.concatenate, post_convert=cy.utils.Arrays)
        outfile = '{}/gp/trials/{}/{}/trials.dict'.format (state.base_dir, state.ana_name, template)
        print ('->', outfile)
        with open (outfile, 'wb') as f:
            pickle.dump (bg, f, -1)

@cli.command ()
@click.option ('--template', default=None, type=str, 
    help='Only calculate for a particular template, default is all')
@click.option ('--nsigma', default=None, type=float, help='Number of sigma to find')
@click.option ('--fit/--nofit', default=False, help = 'Fit the bkg dist to a chi2 or not?')
@click.option ('--verbose/--noverbose', default=False, help = 'Noisy Output')
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def find_gp_n_sig(state, template, nsigma, fit, verbose, inputdir):
    """
    Calculate the Sensitivity or discovery potential once bg and sig files are collected
    Does all galactic plane templates
    """
    ana = state.ana
    flux = []
    def find_n_sig_gp(template, gamma=2.0, beta=0.9, nsigma=None, cutoff=None, verbose=False):
        # get signal trials, background distribution, and trial runner
        if cutoff == None:
            cutoff = np.inf
            cutoff_GeV = np.inf
        else:
            cutoff_GeV = 1e3 * cutoff
        if verbose:
            print(gamma, cutoff)
        if template == 'fermibubbles':
            sig_trials = cy.bk.get_best(sig, 'poisson', 'cutoff', cutoff,  'nsig')
        else:
            sig_trials = cy.bk.get_best(sig, 'poisson',  'nsig')
        b = sig_trials[0.0]['ts']
        if verbose:
            print(b)

        def get_tr(temp, cpus=1):
            gp_conf = cg.get_gp_conf(
                template_str=temp,
                cutoff_GeV=cutoff_GeV,
                base_dir=state.base_dir,
            )
            tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
            return tr

        tr = get_tr(template)
        if nsigma !=None:
            #print('sigma = {}'.format(nsigma))
            if fit:
                ts = cy.dists.Chi2TSD(b).isf_nsigma(nsigma)
            else:
                ts = cy.dists.TSD(b).isf_nsigma(nsigma)
        else:
            #print('Getting sensitivity')
            ts = cy.dists.TSD(b).median()
        if verbose:
            print(ts)

        result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=verbose, trials=sig_trials)
        if template == 'pi0':
            flux = tr.to_E2dNdE(result, E0 = 100 , unit = 1e3)
        elif template == 'fermibubbles':
            flux  = tr.to_dNdE(result, 
                E0 = 1 , unit = 1e3, flux = cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV))
        else:
            flux = tr.to_model_norm(result)
        # return flux
        if verbose:
            print(ts, beta, result['n_sig'], flux)
        return flux

    if nsigma:
        beta = 0.5
    else:
        beta = 0.9
    if template:
        templates = [template]
    else:
        templates = ['fermibubbles', 'pi0', 'kra5', 'kra50']
    for template in templates:
        if inputdir:
            indir = inputdir
        else:
            indir = state.base_dir + '/gp/trials/{}/{}/'.format(state.ana_name, template)
        base_dir = state.base_dir + '/gp/trials/{}/{}/'.format(state.ana_name, template)
        sigfile = '{}/trials.dict'.format (indir)
        sig = np.load (sigfile, allow_pickle=True)
        print('Template: {}'.format(template))
        if template == 'fermibubbles':
            for cutoff in [50,100,500,np.inf]:
                f = find_n_sig_gp(template, beta=beta, nsigma=nsigma, cutoff=cutoff, verbose=verbose)
                flux.append(f) 
                print('Cutoff: {} TeV'.format(cutoff))
                print('Flux: {:.8}'.format(f))    
            print(flux)
            if nsigma:
                np.save(base_dir + '/{}_dp_{}sigma_flux.npy'.format(template, nsigma), flux)
            else:
                np.save(base_dir + '/{}_sens_flux.npy'.format(template), flux)

        else:
            f = find_n_sig_gp(template, nsigma=nsigma,beta =beta, cutoff=cutoff, verbose=verbose)
            print('Flux: {:.8}'.format(f))     
            if nsigma:
                np.save(base_dir + '/{}_dp_{}sigma_flux.npy'.format(template, nsigma), f)
            else:
                np.save(base_dir + '/{}_sens_flux.npy'.format(template), f)


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
    catalog = catalog.lower()
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

@cli.command()
@click.option('--n-trials', default=10000, type=int)
@click.option ('--catalog',   default='snr' , type=str, help='Stacking Catalog, SNR, PWN or UNID')
@click.option ('--gamma', default=2.0, type=float, help = 'Spectrum to Inject')
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--nsigma', default=None, type=float)
@pass_state
def do_stacking_sens (
        state, n_trials, gamma, cutoff, catalog,
        seed, cpus, nsigma,logging=True):
    """
    Do senstivity calculation for stacking catalog.  Useful for quick numbers, not for
    analysis level numbers of trials
    """

    catalog = catalog.lower()
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
    out_dir = cy.utils.ensure_dir ('{}/stacking/sens/{}/'.format (state.base_dir, catalog))

    def get_tr(src, gamma, cpus):
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr

    tr = get_tr(src, gamma, cpus)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()
    bg = cy.dists.Chi2TSD(tr.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))
    if nsigma:
        sens = tr.find_n_sig(
                        bg.isf_nsigma(nsigma), 
                        0.5, #percent above threshold (0.5 for dp)
                        n_sig_step=25,
                        batch_size = n_trials / 3, 
                        tol = 0.02,
                        seed =seed)
    else:
        sens = tr.find_n_sig(
                        bg.median(), 
                        0.9, #percent above threshold (0.9 for sens)
                        n_sig_step=5,
                        batch_size = n_trials / 3, 
                        tol = 0.02,
                        seed = seed)
    sens['flux'] = tr.to_E2dNdE(sens['n_sig'], E0=100, unit=1e3)
    print ('Finished sens at {} ...'.format (t1))
    print (t1 - t0, 'elapsed.')
    print(sens['flux'])
    flush ()
    if nsigma == 0:
        out_file = out_dir + 'sens.npy'
    else: 
        out_file = out_dir + 'dp{}.npy'.format(nsigma)
    np.save(out_file, sens)

@cli.command ()
@click.option ('--dist/--nodist', default=False)
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def collect_stacking_bg (state, dist, inputdir):
    """
    Collect all background trials for stacking into one dictionary for calculation of sensitvity
    """
    bg = {'cat': {}}
    cats = ['snr' , 'pwn', 'unid']
    for cat in cats:
        if inputdir:
            bg_dir = inputdir
        else:
            bg_dir = cy.utils.ensure_dir ('{}/stacking/trials/{}/catalog/{}/bg/'.format (
                state.base_dir, state.ana_name, cat))
        print(bg_dir)
        print ('\r{} ...'.format (cat) + 10 * ' ', end='')
        flush ()
        if dist:
            bg = cy.bk.get_all (
                bg_dir, 'trials*npy',
                merge=np.concatenate, post_convert=(lambda x: cy.dists.Chi2TSD (cy.utils.Arrays (x))))
        else:
            bg = cy.bk.get_all (
                bg_dir, 'trials*npy',
                merge=np.concatenate, post_convert=cy.utils.Arrays )

        print ('\rDone.              ')
        flush ()
        if dist:
            outfile = '{}/stacking/{}_bg_chi2.dict'.format (
                state.base_dir,  cat)
        else:
            outfile = '{}/stacking/{}_bg.dict'.format (
                state.base_dir, cat)
        print ('->', outfile)
        with open (outfile, 'wb') as f:
            pickle.dump (bg, f, -1)

@cli.command ()
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@pass_state
def collect_stacking_sig (state, inputdir):
    """
    Collect all signal trials for stacking into one dictionary for calculation of sensitvity
    """
    cats = 'snr pwn unid'.split ()
    for cat in cats:
        if inputdir:
            sig_dir = inputdir
        else:
            sig_dir = '{}/stacking/trials/{}/catalog/{}/poisson'.format (
                state.base_dir, state.ana_name, cat)
        sig = cy.bk.get_all (
            sig_dir, '*.npy', merge=np.concatenate, post_convert=cy.utils.Arrays)
        outfile = '{}/stacking/{}_sig.dict'.format (
            state.base_dir,  cat)
        with open (outfile, 'wb') as f:
            pickle.dump (sig, f, -1)
        print ('->', outfile)

@cli.command ()
@click.option ('--nsigma', default=None, type=float, help='Number of sigma to find')
@click.option ('--fit/--nofit', default=False, help='Use chi2fit')
@click.option('--inputdir', default=None, type=str, help='Option to Define an input directory outside of default')
@click.option ('--verbose/--noverbose', default=False, help = 'Noisy Output')
@pass_state
def find_stacking_n_sig(state, nsigma, fit, inputdir, verbose):
    """
    Calculate the Sensitvity or discovery potential once bg and sig files are collected
    Does all stacking catalogs
    """
    cutoff = None
    this_dir = os.path.dirname(os.path.abspath(__file__))
    ana = state.ana

    def find_n_sig_cat(src, gamma=2.0, beta=0.9, nsigma=None, cutoff=None, verbose=False):
        # get signal trials, background distribution, and trial runner
        if cutoff == None:
            cutoff = np.inf
            cutoff_GeV = np.inf
        else:
            cutoff_GeV = 1e3 * cutoff
        if verbose:
            print(gamma, cutoff)
        sig_trials = cy.bk.get_best(sig,  'gamma', gamma, 'cutoff_TeV', 
            cutoff, 'nsig')
        b = bg
        if verbose:
            print(b)
        conf = cg.get_ps_conf(src=src, gamma=gamma, cutoff_GeV=cutoff_GeV)
        tr = cy.get_trial_runner(ana=ana, conf=conf)
            # determine ts threshold
        if nsigma !=None:
            #print('sigma = {}'.format(nsigma))
            if fit:
                ts = cy.dists.Chi2TSD(b).isf_nsigma(nsigma)
            else:
                ts = cy.dists.TSD(b).isf_nsigma(nsigma)
        else:
            #print('Getting sensitivity')
            ts = cy.dists.Chi2TSD(b).median()
        if verbose:
            print(ts)

        # include background trials in calculation
        trials = {0: b}
        trials.update(sig_trials)

        result = tr.find_n_sig(ts, beta, max_batch_size=0, logging=verbose, trials=trials)
        flux = tr.to_E2dNdE(result['n_sig'], E0=100, unit=1e3)
        # return flux
        if verbose:
            print(ts, beta, result['n_sig'], flux)
        return flux 
    fluxs = []
    if nsigma:
        beta = 0.5
    else:
        beta = 0.9
    cats = ['snr', 'pwn', 'unid']
    for cat in cats:
        if inputdir:
            indir = iputdir
        else:
            indir = state.base_dir + '/stacking/'
        base_dir = state.base_dir + '/stacking/'
        sigfile = '{}/{}_sig.dict'.format (indir, cat)
        sig = np.load (sigfile, allow_pickle=True)
        bgfile = '{}/{}_bg.dict'.format (indir, cat)
        bg = np.load (bgfile, allow_pickle=True)
        print('CATALOG: {}'.format(cat))
        srcs= np.load('{}/catalogs/{}_ESTES_12.pickle'.format(this_dir, cat), allow_pickle=True)
        src = cy.utils.Sources(ra = srcs['ra_deg'], dec=srcs['dec_deg'], deg=True)
        for gamma in sig['gamma'].keys():
            print ('Gamma: {}'.format(gamma))
            f = find_n_sig_cat(src, gamma=gamma, beta=beta, nsigma=nsigma, cutoff=cutoff, verbose=verbose)
            print('Sensitvity Flux: {:.8}'.format(f))     
            fluxs.append(f)
            if nsigma:
                np.save(base_dir + '/stacking_{}_dp_{}sigma_flux_E{}.npy'.format(cat, nsigma, int(gamma * 100)), fluxs)
            else:
                np.save(base_dir + '/stacking_{}_sens_flux_E{}.npy'.format(cat, int(gamma * 100)), fluxs)


@cli.command()
@click.option('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option('-n', '--n-sig', default=0, type=float,
              help='Number of signal events to inject')
@click.option('--nside', default=128, type=int)
@click.option('--cpus', default=1, type=int)
@click.option('--seed', default=None, type=int)
@click.option('--poisson/--nopoisson', default=True,
              help='toggle possion weighted signal injection')
@click.option('--gamma', default=2.0, type=float,
              help='Gamma for signal injection.')
@click.option('--fit/--nofit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@pass_state
def do_sky_scan_trials(
        state, poisson, dec_deg, nside, n_sig, cpus, seed, gamma, fit):
    """
    Scan each point in the sky in a grid of pixels
    """

    if seed is None:
        seed = int(time.time() % 2**32)
    random = cy.utils.get_random(seed)
    print('Seed: {}'.format(seed))
    dec = np.radians(dec_deg)
    sindec = np.sin(dec)
    base_dir = state.base_dir + '/ps/trials/DNNC'
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
        bgs = np.load(bgfile, allow_pickle=True)['dec']
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
        bg_trials = np.load(bgfile, allow_pickle=True)['dec']
        bgs = {key: cy.dists.TSD(trials) for key, trials in bg_trials.items()}

    def ts_to_p(dec, ts):
        return cy.dists.ts_to_p(bgs, np.degrees(dec), ts, fit=fit)

    t0 = now()
    ana = state.ana
    conf = cg.get_ps_conf(src=None, gamma=gamma)
    conf.pop('src')
    conf.update({
        'ana': ana,
        'mp_cpus': cpus,
        'extra_keep': ['energy'],
    })

    inj_src = cy.utils.sources(ra=0, dec=dec_deg, deg=True)
    inj_conf = {
        'src': inj_src,
        'flux': cy.hyp.PowerLawFlux(gamma),
    }

    sstr = cy.get_sky_scan_trial_runner(conf=conf, inj_conf=inj_conf,
                                        min_dec=np.radians(-80),
                                        max_dec=np.radians(80),
                                        mp_scan_cpus=cpus,
                                        nside=nside, ts_to_p=ts_to_p)
    print('Doing one Scan with nsig = {}'.format(n_sig))
    trials = sstr.get_one_scan(n_sig, poisson=poisson, logging=True, seed=seed)

    base_out = '{}/skyscan/trials/{}/nside/{:04d}'.format(
        state.base_dir, state.ana_name, nside)
    if n_sig:
        out_dir = cy.utils.ensure_dir(
            '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
                base_out,
                'poisson' if poisson else 'nonpoisson',
                'fit' if fit else 'nofit',
                gamma,  dec_deg, n_sig))
    else:
        out_dir = cy.utils.ensure_dir('{}/bg/{}'.format(
            base_out, 'fit' if fit else 'nofit'))
    out_file = '{}/scan_seed_{:010d}.npy'.format(out_dir,  seed)
    print ('-> {}'.format(out_file))
    np.save(out_file, trials)


@cli.command()
@click.option('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option('-n', '--n-sig', default=0, type=float,
              help='Number of signal events to inject')
@click.option('--nside', default=128, type=int)
@click.option('--poisson/--nopoisson', default=True,
              help='toggle possion weighted signal injection')
@click.option('--gamma', default=2.0, type=float,
              help='Gamma for signal injection.')
@click.option('--overwrite/--nooverwrite', default=False,
              help='If True, existing files will be overwritten')
@click.option('--fit/--nofit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@click.option('--inputfit/--noinputfit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@pass_state
def recalculate_sky_scan_trials(
        state, poisson, dec_deg, nside, n_sig, gamma, fit, inputfit,
        overwrite):
    """
    Recalculate previous sky scan result based on given background trials.

    This can be used to update old sky-scans if more background trials become
    available at each declination value, or if one wants to change from
    `--fit` (estimate via Chi2 Fit) to `--nofit` (use trials directly) and
    vice versa.
    """

    dec = np.radians(dec_deg)
    print('Loading background trials...')
    base_dir = state.base_dir + '/ps/trials/DNNC'
    if fit:
        bgfile = '{}/bg_chi2.dict'.format(base_dir)
        bgs = np.load(bgfile, allow_pickle=True)['dec']
    else:
        bgfile = '{}/bg.dict'.format(base_dir)
        bg_trials = np.load(bgfile, allow_pickle=True)['dec']
        bgs = {key: cy.dists.TSD(trials) for key, trials in bg_trials.items()}

    def ts_to_p(dec, ts):
        return cy.dists.ts_to_p(bgs, np.degrees(dec), ts, fit=fit)

    # get input and output directories
    base_out = '{}/skyscan/trials/{}/nside/{:04d}'.format(
        state.base_dir, state.ana_name, nside)
    if n_sig:
        input_dir = '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
            base_out,
            'poisson' if poisson else 'nonpoisson',
            'fit' if inputfit else 'nofit',
            gamma,  dec_deg, n_sig)
        out_dir = cy.utils.ensure_dir(
            '{}/{}/{}/gamma/{:.3f}/dec/{:+08.3f}/nsig/{:08.3f}'.format(
                base_out,
                'poisson' if poisson else 'nonpoisson',
                'fit' if fit else 'nofit',
                gamma,  dec_deg, n_sig))
    else:
        input_dir = '{}/bg/{}'.format(base_out, 'fit' if inputfit else 'nofit')
        out_dir = cy.utils.ensure_dir('{}/bg/{}'.format(
            base_out, 'fit' if fit else 'nofit'))

    # collect sky scans that will be recalculated
    print('Collecting input files...')
    input_files = sorted(glob.glob(os.path.join(input_dir, 'scan_seed_*.npy')))

    print('Found {} files. Recalculating p-values...'.format(len(input_files)))
    for input_file in input_files:

        # load and recalculate scan
        scan = np.load(input_file, allow_pickle=True)
        new_scan = utils.recalculate_scan(scan=scan, ts_to_p=ts_to_p)

        out_file = os.path.join(out_dir,  os.path.basename(input_file))

        if not overwrite and os.path.exists(out_file):
            msg = 'File {} already exists. To overwrite, pass `--overwrite`.'
            raise IOError(msg.format(out_file))

        print('-> {}'.format(out_file))
        np.save(out_file, new_scan)


@cli.command()
@pass_state
def collect_sky_scan_trials_bg(state):
    """
    Collect hottest p-value from background sky scan trials
    """

    base_dir = '{}/skyscan/trials/{}/'.format(state.base_dir, state.ana_name)

    # pre-calculate mask for northern pixels for given nside
    nside_dirs = glob.glob(os.path.join(base_dir, 'nside', '*'))
    nside_list = [int(os.path.basename(nside_dir)) for nside_dir in nside_dirs]
    mask_north_dict = utils.get_mask_north_dict(nside_list=nside_list)

    trials = cy.bk.get_all(
        base_dir, 'scan_seed_*.npy',
        pre_convert=utils.extract_hottest_p_value,
    )

    with open('{}sky_scan_bg.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


@cli.command()
@click.option ('--sourcenum', default=1, type=int, help='what source in the list')
@click.option ('--n-trials', default=1000, type=int, help='number of trials to run')
@click.option ('--cpus', default=1, type=int, help='ncpus')
@click.option ('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_bkg_trials_sourcelist (
        state, sourcenum, n_trials, cpus, seed, logging=True):
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
    trials = tr.get_many_fits(n_trials, seed=seed)
    t1 = now ()
    flush ()
    out_dir = cy.utils.ensure_dir ('{}/ps/correlated_trials/bg/source_{}'.format (
        state.base_dir, sourcenum))
    out_file = '{}/bkg_trials_{}_seed_{}.npy'.format (
        out_dir, n_trials, seed)
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
    assert len(bgs) == len(sourcelist)

    def get_tr(dec, ra, cpus=cpus):
        src = cy.utils.sources(ra=ra, dec=dec, deg=True)
        conf = cg.get_ps_conf(src=src, gamma=2.0)
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr
    print('Getting trial runners')
    trs = [get_tr(d,r) for d,r in zip(decs, ras)]
    assert len(trs) == len(bgs)

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
    t1 = now ()
    flush ()
    out_dir = cy.utils.ensure_dir ('{}/ps/correlated_trials/correlated_bg/'.format (
        state.base_dir, state.ana_name))
    out_file = '{}/correlated_trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save(out_file, trials.as_array)


@cli.command()
@pass_state
def collect_correlated_trials_sourcelist(state):
    """
    Collect all the correlated MultiTrialRunner background trials from the
    do-correlated-trials-sourcelist into one list. These will be used to
    trial-correct the p-value of the hottest source in the source list.
    """
    base_dir = '{}/ps/correlated_trials/'.format(state.base_dir)
    trials = cy.bk.get_all(
        base_dir + 'correlated_bg/', 'correlated_trials_*.npy',
        merge=np.concatenate,
        post_convert=(lambda x: cy.utils.Arrays(x)),
    )

    # find and add hottest source
    # shape: [n_trials, n_sources]
    mlog10ps = np.stack([
        trials[k] for k in trials.keys() if k[:8] == 'mlog10p_'], axis=1)

    trials['ts'] = np.max(mlog10ps, axis=1)
    trials['idx_hottest'] = np.argmax(mlog10ps, axis=1)

    with open('{}correlated_bg.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


@cli.command()
@click.option(
    '--n-trials', default=10000, type=int, help='Number of trials to run')
@click.option('--cpus', default=1, type=int, help='ncpus')
@click.option('--seed', default=None, type=int, help='Trial injection seed')
@pass_state
def do_correlated_trials_fermibubbles(
        state, n_trials, cpus, seed,  logging=True):
    """Correlated trials for Fermibubbles

    Use MTR for correlated background trials evaluating for each cutoff
    """
    cutoffs = [50, 100, 500]
    if seed is None:
        seed = int(time.time() % 2**32)

    t0 = now()
    ana = state.ana
    print('Loading Backgrounds')
    fermi_dir = '{}/gp/trials/{}/fermibubbles'.format(
        state.base_dir, state.ana_name)
    trials = np.load('{}/trials.dict'.format(fermi_dir), allow_pickle=True)

    # collect list of background trials for each cutoff
    bgs = [
        cy.dists.TSD(trials['poisson']['cutoff'][cutoff]['nsig'][0.0])
        for cutoff in cutoffs
    ]

    def get_tr(temp, cutoff):
        cutoff_GeV = cutoff * 1e3
        gp_conf = cg.get_gp_conf(
            template_str=temp, cutoff_GeV=cutoff_GeV, base_dir=state.base_dir)
        tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus=cpus)
        return tr

    print('Getting trial runners')
    trs = [get_tr('fermibubbles', cutoff) for cutoff in cutoffs]
    assert len(trs) == len(bgs)

    tr_inj = trs[0]
    multr = cy.trial.MultiTrialRunner(
        ana,
        # bg+sig injection trial runner (produces trials)
        tr_inj,
        # llh test trial runners (perform fits given trials)
        trs,
        # background distributions
        bgs=bgs,
        # use multiprocessing
        mp_cpus=cpus,
    )
    trials = multr.get_many_fits(n_trials)
    t1 = now()
    flush()
    out_dir = cy.utils.ensure_dir('{}/correlated_trials/correlated_bg'.format(
        fermi_dir))
    out_file = '{}/correlated_trials_{:07d}__seed_{:010d}.npy'.format(
        out_dir, n_trials, seed)
    print ('-> {}'.format(out_file))
    np.save(out_file, trials.as_array)


@cli.command()
@pass_state
def collect_correlated_trials_fermibubbles(state):
    """
    Collect all the correlated MultiTrialRunner background trials from the
    do-correlated-trials-fermibubbles into one list. These will be used to
    trial-correct the p-value of the most significant Fermibubble
    energy cutoff.
    """
    base_dir = '{}/gp/trials/{}/fermibubbles/correlated_trials'.format(
        state.base_dir, state.ana_name)
    trials = cy.bk.get_all(
        base_dir + '/correlated_bg/', 'correlated_trials_*.npy',
        merge=np.concatenate,
        post_convert=(lambda x: cy.utils.Arrays(x)),
    )

    # find and add most significant cutoff
    # shape: [n_trials, n_cutoffs]
    mlog10ps = np.stack([
        trials[k] for k in trials.keys() if k[:8] == 'mlog10p_'], axis=1)

    trials['ts'] = np.max(mlog10ps, axis=1)
    trials['idx_hottest'] = np.argmax(mlog10ps, axis=1)

    with open('{}/correlated_bg.npy'.format(base_dir), 'wb') as f:
        pickle.dump(trials, f, -1)


if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
