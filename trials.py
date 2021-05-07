#!/usr/bin/env python

from __future__ import print_function
import csky as cy
from csky import coord
import numpy as np
import pickle, datetime, socket
from submitter import Submitter
import histlite as hl
import astropy
#from icecube import astro
now = datetime.datetime.now
import matplotlib.pyplot as plt
import click, sys, os, time
flush = sys.stdout.flush

hostname = socket.gethostname()
print('Hostname: {}'.format(hostname))
if 'condor00' in hostname or 'cobol' in hostname or 'gpu' in hostname:
    print('Using UMD')
    repo = cy.selections.Repository(local_root='/data/i3store/users/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses/DNNC_11yr')
    job_basedir = '/data/i3home/ssclafani/submitter_logs'
else:
    repo = cy.selections.Repository(local_root='/data/user/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses/DNNC_11_yrs')
    ana_dir = '{}/ana'.format (base_dir)
    job_basedir = '/scratch/ssclafani/' 

class State (object):
    def __init__ (self, ana_name, ana_dir, save, base_dir, job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = ana_name, ana_dir, save, job_basedir
        self.base_dir = base_dir
        self._ana = None

    @property
    def ana (self):
        if self._ana is None:
            repo.clear_cache()
            specs = cy.selections.DNNCascadeDataSpecs.DNNC_11yr
            ana = cy.analysis.Analysis (repo, specs)#r=self.ana_dir)
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
@click.option ('-a', '--ana', 'ana_name', default='ECAS', help='Dataset title')
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
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--gamma', default=2.0, type=float)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def do_all_sky_sens ( state, n_trials, gamma, n_sig, poisson, seed, cpus, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    ana = state.ana
    print('Doing Trials with : ')
    print('Muon BDT Cut : {}'.format(state.mucut))
    print('Cascade BDT Cut :{}'.format(state.ccut))
    print('Energy Threshold Cut :{}GeV'.format(state.ethresh))
    print('AngRes Cut: {}deg'.format( state.angrescut ))
    #dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir))
    sindecs = np.arange(-1.,1.01,.05)
    sindecs[-1] = .98
    sindecs[0] = -.98
    def get_PS_sens(sindec, n_trials=n_trials, gamma=gamma, mp_cpu=cpus):
        src = cy.utils.sources(0, np.arcsin(sindec), deg=False)
        tr = cy.get_trial_runner(src=src, ana=ana, flux=cy.hyp.PowerLawFlux(gamma), mp_cpus=mp_cpu)
        print('Performing BG Trails at RA: {}, DEC: {}'.format(src.ra_deg, src.dec_deg))
        bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials, mp_cpus=mp_cpu))
        sens = tr.find_n_sig(
            # ts, threshold
            bg.median(),
            # beta, fraction of trials which should exceed the threshold
            0.9,
            # n_inj step size for initial scan
            n_sig_step=6,
            # this many trials at a time
            batch_size=2500,
            # tolerance, as estimated relative error
            tol=.025,
            first_batch_size = 250,
            mp_cpus=mp_cpu
        )
        sens['flux'] = tr.to_E2dNdE (sens['n_sig'], E0=100, unit=1e3)
        print(sens['flux'])
        return sens

    cy.utils.ensure_dir(ana_dir + '/performance/ECAS_loose_scan')
    t0 = now ()

    print ('Beginning calculation at {} ...'.format (t0))
    flush ()
    sens_hacs = [get_PS_sens (sindec, gamma=gamma, n_trials=n_trials) for sindec in sindecs]
    
    sens_hacs_flux = np.array ([s['flux'] for s in sens_hacs])
    
    print(sens_hacs_flux)
    repo.save_performance(sindecs, sens_hacs_flux, 'ECAS_loose_scan/ECAS_sens_E{}_mu_{}_angres_{}_E_{}'.format(int(gamma*100 ), state.mucut, state.angrescut, state.ethresh) ,
            'mucut:{} angrescut:{} Ethresh:{} gamma: {}'.format(state.mucut, state.angrescut, state.ethresh, gamma))
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))


@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--gamma', default=2.0, type=float)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--nn/--nonn', default=True)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@pass_state
def do_ps_sens ( 
        state, n_trials, poisson,gamma, dec_deg, seed, 
        cpus, model_name, additionalpdfs, nn, cutoff, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    ana = state.ana
    sindec = np.sin(np.radians(dec_deg))
    ratio_model_name = model_name
    def get_PS_sens(sindec, n_trials=n_trials, gamma=gamma, mp_cpu=cpus, additionalpdfs=additionalpdfs):
        def get_tr(sindec, gamma, cpus, additionalpdfs):
            src = cy.utils.sources(0, np.arcsin(sindec), deg=False)
            cutoff_GeV = cutoff * 1e3
            if nn:
                from csky import models
                import pickle
                from csky.models import serialization
                from csky.models.serialization import load_model
                from csky import models
                import tensorflow as tf
                additionalpdfs = 'sigma' #need to enable sigma pdf if you use the NN

                print('Using NN description of Signal and BG')
                def get_model_name(**kwargs):
                    model_name = 'model'
                    for key in sorted(kwargs.keys()):
                        value = kwargs[key]
                        if isinstance(value, float):
                            model_name += '_{}{:3.2e}'.format(key, value)
                        else:
                            model_name += '_{}{}'.format(key, value)
                    return model_name
                model_base_dir ='/data/i3store/users/ssclafani/models'
                cut_settings = {
                    'mucut' : 5e-3,
                    'ccut' : 0.3,
                    'ethresh' : 500,
                    'angrescut' : float(np.deg2rad(30)),
                    'angfloor' : float(np.deg2rad(0.5))
                    }
                model_dir = '{}/sigma_pdf_test/{}'.format(
                    model_base_dir, get_model_name(**cut_settings))    

                observables = ['log10energy']
                conditionals = ['sindec', 'sigma', 'gamma']
                observables = sorted(observables)
                conditionals = sorted(conditionals)
                obs_str = observables[0].replace('src->', 'src')        
                for obs in observables[1:]:
                    obs_str += '_' + obs.replace('src->', 'src')

                conditionals = sorted(conditionals)
                if len(conditionals) > 0:
                    cond_str = conditionals[0].replace('src->', 'src')
                else:
                    cond_str = 'None'
                for cond in conditionals[1:]:
                    cond_str += '_' + cond.replace('src->', 'src')

                ratio_model_path = '{}/{}/{}/ratio_model_interp/{}'.format(
                         model_dir, obs_str, cond_str, ratio_model_name)
                ratio_model, info  = load_model(ratio_model_path)  
                print('features:', ratio_model.features)
                print('parameters:', ratio_model.parameters)
                print('observables:', ratio_model.observables)
                print('conditionals:', ratio_model.conditionals)

                #Failsafe Checks!!
                trained_cut_settings = info['info']['cut_settings']
                if trained_cut_settings['angrescut'] != np.deg2rad(30):
                    print(trained_cut_settings['angrescut'])
                    raise Exception('Angres cut doesnt Match!!')
                if trained_cut_settings['ccut'] != 0.3:
                    print(trained_cut_settings['ccut'])
                    raise Exception('CascadeBDT cut doesnt Match!!')
                if trained_cut_settings['mucut'] != 5e-3:
                    print(trained_cut_settings['mucut'])                               
                    raise Exception('MuonBDT cut doesnt Match!!')
                if trained_cut_settings['ethresh'] != 500:
                    print(trained_cut_settings['ethresh'])
                    raise Exception('Ethresh cut doesnt Match!!')


                ratio_model, info  = load_model(ratio_model_path)
                print('features:', ratio_model.features)
                print('parameters:', ratio_model.parameters)
                print('observables:', ratio_model.observables)
                print('conditionals:', ratio_model.conditionals) 
                conf =  {
                    'src' : src,
                    cy.pdf.GenericPDFRatioModel: dict(
                                func = ratio_model,
                                features = ratio_model.features,
                                fits = dict(gamma=np.r_[1, 1:4.01:.125, 4]),
                            ),
                    'flux' : cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV),
                    'update_bg': True,
                    'energy' : False #If using the NN energy needs to be set to False, the energy term is in the NN
                    }   
            else:
                conf = {
                    'src' : src,
                    'flux' : cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV),
                    'update_bg': True, 
                    }
            if additionalpdfs == 'sigma':
                print('Space * E * Sigma')   
                conf[cy.pdf.SigmaPDFRatioModel] = dict(
                            hkw=dict(bins=( np.linspace(-1,1,20),  np.linspace(0, np.deg2rad(30), 20))), 
                            features=['sindec', 'sigma'],
                            normalize_axes = ([1])) 
            tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
            return tr, src
        tr, src = get_tr(sindec, gamma, cpus, additionalpdfs)
        print('Performing BG Trails at RA: {}, DEC: {}'.format(src.ra_deg, src.dec_deg))
        bg = cy.dists.Chi2TSD(tr.get_many_fits(n_trials, mp_cpus=cpus))
        sens = tr.find_n_sig(
            # ts, threshold
            bg.median(),
            # beta, fraction of trials which should exceed the threshold
            0.9,
            # n_inj step size for initial scan
            n_sig_step=6,
            # this many trials at a time
            batch_size=2500,
            # tolerance, as estimated relative error
            tol=.025,
            first_batch_size = 250,
            mp_cpus=mp_cpu
        )
        sens['flux'] = tr.to_E2dNdE (sens['n_sig'], E0=100, unit=1e3)
        print(sens['flux'])
        return sens

    t0 = now ()
    print ('Beginning calculation at {} ...'.format (t0))
    flush ()
    sens = get_PS_sens (sindec, gamma=gamma, n_trials=n_trials) 
    
    sens_flux = np.array(sens['flux'])
    if nn:
        out_dir = cy.utils.ensure_dir('{}/E{}/NN/dec/{:+08.3f}/'.format(
            state.base_dir, int(gamma*100),   dec_deg))
    else:
        out_dir = cy.utils.ensure_dir('{}/E{}/NoNN/dec/{:+08.3f}/'.format(
            state.base_dir, int(gamma*100),  dec_deg))

    out_file = out_dir + 'sens.npy'
    print(sens_flux)
    np.save(out_file, sens_flux)
    t1 = now ()
    print ('Finished sens at {} ...'.format (t1))

@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--gamma', default=2.0, type=float)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--nn/--nonn', default=True)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def do_ps_trials ( 
        state, dec_deg, n_trials, gamma, cutoff, 
        model_name, additionalpdfs, nn,  n_sig, 
        poisson, seed, cpus, logging=True):

    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    dec = np.radians(dec_deg)
    sindec = np.sin(dec)
    ana = state.ana
    ratio_model_name = model_name
    src = cy.utils.Sources(dec=dec, ra=0)
    cutoff_GeV = cutoff * 1e3
    dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir, dec_deg))
    def get_tr(sindec, gamma, cpus, additionalpdfs):
        src = cy.utils.sources(0, np.arcsin(sindec), deg=False)
        cutoff_GeV = cutoff * 1e3
        if nn:
            from csky import models
            import pickle
            from csky.models import serialization
            from csky.models.serialization import load_model
            from csky import models
            import tensorflow as tf
            print('Using NN description of Signal and BG')
            
            ###need to add the sigma PDF
            additionalpdfs = 'sigma'

            def get_model_name(**kwargs):
                model_name = 'model'
                for key in sorted(kwargs.keys()):
                    value = kwargs[key]
                    if isinstance(value, float):
                        model_name += '_{}{:3.2e}'.format(key, value)
                    else:
                        model_name += '_{}{}'.format(key, value)
                return model_name
            model_base_dir ='/data/i3store/users/ssclafani/models'
            cut_settings = {
                'mucut' : 5e-3,
                'ccut' : 0.3,
                'ethresh' : 500,
                'angrescut' : float(np.deg2rad(30)),
                'angfloor' : float(np.deg2rad(0.5))
                }
            model_dir = '{}/sigma_pdf_test/{}'.format(
                model_base_dir, get_model_name(**cut_settings))    

            observables = ['log10energy']
            conditionals = ['sindec', 'sigma', 'gamma']
            observables = sorted(observables)
            conditionals = sorted(conditionals)
            obs_str = observables[0].replace('src->', 'src')        
            for obs in observables[1:]:
                obs_str += '_' + obs.replace('src->', 'src')

            conditionals = sorted(conditionals)
            if len(conditionals) > 0:
                cond_str = conditionals[0].replace('src->', 'src')
            else:
                cond_str = 'None'
            for cond in conditionals[1:]:
                cond_str += '_' + cond.replace('src->', 'src')

            ratio_model_path = '{}/{}/{}/ratio_model_interp/{}'.format(
                     model_dir, obs_str, cond_str, ratio_model_name)
            ratio_model, info  = load_model(ratio_model_path)  
            print('features:', ratio_model.features)
            print('parameters:', ratio_model.parameters)
            print('observables:', ratio_model.observables)
            print('conditionals:', ratio_model.conditionals)

            #Failsafe Checks!!
            trained_cut_settings = info['info']['cut_settings']
            if trained_cut_settings['angrescut'] != np.deg2rad(30):
                print(trained_cut_settings['angrescut'])
                raise Exception('Angres cut doesnt Match!!')
            if trained_cut_settings['ccut'] != 0.3:
                print(trained_cut_settings['ccut'])
                raise Exception('CascadeBDT cut doesnt Match!!')
            if trained_cut_settings['mucut'] != 5e-3:
                print(trained_cut_settings['mucut'])                               
                raise Exception('MuonBDT cut doesnt Match!!')
            if trained_cut_settings['ethresh'] != 500:
                print(trained_cut_settings['ethresh'])
                raise Exception('Ethresh cut doesnt Match!!')


            ratio_model, info  = load_model(ratio_model_path)
            print('features:', ratio_model.features)
            print('parameters:', ratio_model.parameters)
            print('observables:', ratio_model.observables)
            print('conditionals:', ratio_model.conditionals) 
            conf =  {
                'src' : src,
                cy.pdf.GenericPDFRatioModel: dict(
                            func = ratio_model,
                            features = ratio_model.features,
                            fits = dict(gamma=np.r_[1, 1:4.01:.125, 4]),
                        ),
                'flux' : cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV),
                'update_bg': True,
                'energy' : False #If using the NN energy needs to be set to False, the energy term is in the NN
                }   
        else:
            conf = {
                'src' : src,
                'flux' : cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV),
                'update_bg': True, 
                }
        if additionalpdfs == 'sigma':
            print('Space * E * Sigma')   
            conf[cy.pdf.SigmaPDFRatioModel] = dict(
                        hkw=dict(bins=( np.linspace(-1,1,20),  np.linspace(0, np.deg2rad(30), 20))), 
                        features=['sindec', 'sigma'],
                        normalize_axes = ([1])) 
        tr = cy.get_trial_runner(ana=ana, conf= conf, mp_cpus=cpus)
        return tr, src
    tr , src = get_tr(sindec, gamma, cpus, additionalpdfs)
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
            '{}/ps/trials/{}/{}/gamma/{:.3f}/cutoff_TeV/{:.0f}/dec/{:+08.3f}/nsig/{:08.3f}'.format (
                state.base_dir, state.ana_name,
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
@click.option ('--fit/--nofit', default=True)
@click.option ('--hist/--nohist', default=False)
@click.option ('--dist/--nodist', default=False)
@click.option ('-n', '--n', default=0, type=int)
@pass_state
def collect_ps_bg (state, fit, hist, dist, n):
    print(dist)
    #bg = {'dec': {}}
    kw = {}
    if hist:
        TSD = cy.dists.BinnedTSD
        suffix = '_hist'
        kw['keep_trials'] = False
    elif fit:
        TSD = cy.dists.Chi2TSD
        suffix = '_chi2'
    elif dist:
        print('no distribution')
        suffix = ''
    else:
        TSD = cy.dists.TSD
        suffix = ''
    outfile = '{}/ps/trials/{}/bg{}.dict'.format (
        state.base_dir, state.ana_name,  suffix)
    bg = {}
    bgs = {}
    dec_degs = np.r_[-89:+89.01:4]
    bg_dir = '{}/ps/trials/{}/bg'.format (
        state.base_dir, state.ana_name)
    #print(bg_dir)
    for dec_deg in dec_degs:
        key = '{:+08.3f}'.format (dec_deg)
        #print ('\r{} ...'.format (key), end='')
        flush ()
        print('{}/dec/{}/'.format(bg_dir, key))
        if dist == False:
            print('no dist') 
            post_convert = (lambda x: cy.utils.Arrays (x))
        else:
            post_convert = (lambda x: TSD (cy.utils.Arrays (x), **kw))
        bgs[float(key)] = cy.bk.get_all (
                '{}/dec/{}/'.format (bg_dir, key), '*.npy',
                merge=np.concatenate, post_convert=post_convert)
    bg['dec'] = bgs
    print ('\rDone.' + 20 * ' ')
    flush ()
    print ('->', outfile)
    with open (outfile, 'wb') as f:
        pickle.dump (bg, f, -1)

@cli.command ()
@pass_state
def collect_ps_sig (state):
    sig_dir = '{}/ps/trials/{}/poisson'.format (state.base_dir, state.ana_name)
    sig = cy.bk.get_all (
        sig_dir, '*.npy', merge=np.concatenate, post_convert=cy.utils.Arrays)
    outfile = '{}/ps/trials/{}/sig.dict'.format (state.base_dir, state.ana_name)
    with open (outfile, 'wb') as f:
        pickle.dump (sig, f, -1)
    print ('->', outfile)


@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--nn/--nonn', default=True)
@pass_state
def do_gp_trials ( 
            state, temp, n_trials, n_sig, 
            poisson, seed, cpus, additionalpdfs, model_name, nn, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
    def get_tr(nn, temp, additionalpdfs):
        if nn:
            print('Using NN description of Signal and BG')
            from csky import models
            import pickle
            from csky.models import serialization
            from csky.models.serialization import load_model

            from csky import models
            import tensorflow as tf
            model_base_dir ='/data/i3store/users/ssclafani/models'
            ratio_model_name = model_name
            def get_model_name(**kwargs):
                model_name = 'model'
                for key in sorted(kwargs.keys()):
                    value = kwargs[key]
                    if isinstance(value, float):
                        model_name += '_{}{:3.2e}'.format(key, value)
                    else:
                        model_name += '_{}{}'.format(key, value)
                return model_name
            cut_settings = {
                 'mucut' : 5e-3,
                 'ccut' : 0.3,
                 'ethresh' : 500,
                 'angrescut' : float(np.deg2rad(30)),
                 'angfloor' : float(np.deg2rad(0.5))
            }
            model_dir = '{}/sigma_pdf_test/{}'.format(
                model_base_dir, get_model_name(**cut_settings))    

            observables = ['log10energy']
            if additionalpdfs == 'sigma_src':
                conditionals = ['srcsindec', 'sindec', 'sigma', 'gamma']
            else:    
                conditionals = ['sindec', 'sigma', 'gamma']
            observables = sorted(observables)
            obs_str = observables[0].replace('src->', 'src')
            for obs in observables[1:]:
                obs_str += '_' + obs.replace('src->', 'src')
                
            conditionals = sorted(conditionals)
            if len(conditionals) > 0:
                cond_str = conditionals[0].replace('src->', 'src')
            else:
                cond_str = 'None'
            for cond in conditionals[1:]:
                cond_str += '_' + cond.replace('src->', 'src')
            ratio_model_path = '{}/{}/{}/ratio_model_interp/{}'.format(
                     model_dir, obs_str, cond_str, ratio_model_name)


            ratio_model, info  = load_model(ratio_model_path)
            print('features:', ratio_model.features)
            print('parameters:', ratio_model.parameters)
            print('observables:', ratio_model.observables)
            print('conditionals:', ratio_model.conditionals) 
             
            #print(info)
            trained_cut_settings = info['info']['cut_settings']
            if trained_cut_settings['angrescut'] != np.deg2rad(30):
                print(trained_cut_settings['angrescut'])
                raise Exception('Angres cut doesnt Match!!')
            if trained_cut_settings['ccut'] != 0.3:
                print(trained_cut_settings['ccut'])
                raise Exception('CascadeBDT cut doesnt Match!!')
            if trained_cut_settings['mucut'] != 5e-3:
                print(trained_cut_settings['mucut'])
                raise Exception('MuonBDT cut doesnt Match!!')
            if trained_cut_settings['ethresh'] != 500:
                print(trained_cut_settings['ethresh'])
                raise Exception('Ethresh cut doesnt Match!!')
            if temp == 'pi0':
                template = repo.get_template ('Fermi-LAT_pi0_map')
                gp_conf = {
                    'template': template,
                    'flux':     cy.hyp.PowerLawFlux(2.5),
                    'fitter_args': dict(gamma=2.5),
                    'sigsub': True,
                    'fast_weight': True,
                    cy.pdf.GenericPDFRatioModel: dict(
                                func = ratio_model,
                                features = ratio_model.features,
                                fits = dict(gamma=np.r_[1, 1:4.01:.125, 4]),
                            ),
                    'update_bg': True,
                    'energy' : False #If using the NN energy needs to be set to False, 
                        }   
            elif 'kra' in temp:
                if temp =='kra5':
                    template, energy_bins = repo.get_template(
                              'KRA-gamma_5PeV_maps_energies', per_pixel_flux=True)
                elif temp =='kra50':
                    template, energy_bins = repo.get_template(
                              'KRA-gamma_50PeV_maps_energies', per_pixel_flux=True)
                gp_conf = {
                  # desired template
                  'template': template,
                  'bins_energy': energy_bins,
                  'fitter_args' : dict(gamma=2.5),
                  'update_bg' : True,
                  'sigsub': True,
                  'energy' : False
                  }
            if additionalpdfs == 'sigma':
                print('Space * E * Sigma')   
                gp_conf[cy.pdf.SigmaPDFRatioModel] = dict(
                            hkw=dict(bins=( np.linspace(-1,1,20),  
                            np.linspace(0, np.deg2rad(30), 20))),
                            features=['sindec', 'sigma'],
                            normalize_axes = ([1])) 
            tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus = cpus)
        else:
            if temp == 'pi0':
                template = repo.get_template ('Fermi-LAT_pi0_map')
                gp_conf = {
                    'template': template,
                    'flux':     cy.hyp.PowerLawFlux(2.5),
                    'fitter_args': dict(gamma=2.5),
                    'sigsub': True,
                    'fast_weight': True,
                    'dir': cy.utils.ensure_dir('{}/templates/pi0'.format(ana_dir))}
            elif temp =='kra5':
                template, energy_bins = repo.get_template(
                          'KRA-gamma_5PeV_maps_energies', per_pixel_flux=True)

                gp_conf = {
                    # desired template
                    'template': template,
                    'bins_energy': energy_bins,
                    'fitter_args' : dict(gamma=2.5),
                    'update_bg' : True,
                    'sigsub': True,
                    #'dir': cy.utils.ensure_dir('{}/templates/kra5'.format(ana_dir))
                    }
            if additionalpdfs == 'sigma':
                print('Space * E * Sigma')   
                gp_conf[cy.pdf.SigmaPDFRatioModel] = dict(
                            hkw=dict(bins=( np.linspace(-1,1,20),  np.linspace(0,
                            np.deg2rad(30), 20))),
                            features=['sindec', 'sigma'],
                            normalize_axes = ([1])) 
            tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus = cpus)
        return tr
    tr = get_tr(nn, temp, additionalpdfs)
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
    out_dir = cy.utils.ensure_dir (
        '{}/trials/{}/gp/{}/{}/nsig/{:08.3f}'.format (
            state.base_dir, state.ana_name,
            'poisson' if poisson else 'nonpoisson',
            temp,
            n_sig))
    out_file = '{}/trials_{:07d}__seed_{:010d}.npy'.format (
        out_dir, n_trials, seed)
    print ('-> {}'.format (out_file))
    np.save (out_file, trials.as_array)

@cli.command()
@click.argument('temp')
@click.option('--n-trials', default=1000, type=int)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('--nn/--nonn', default=True)
@pass_state
def do_gp_sens ( state, temp, n_trials, additionalpdfs, model_name, seed, cpus, nn, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))

    def get_tr(nn, temp, additionalpdfs):
        if nn:
            print('Using NN description of Signal and BG')
            from csky import models
            import pickle
            from csky.models import serialization
            from csky.models.serialization import load_model

            from csky import models
            import tensorflow as tf
            model_base_dir ='/data/i3store/users/ssclafani/models'
            ratio_model_name = model_name
            def get_model_name(**kwargs):
                model_name = 'model'
                for key in sorted(kwargs.keys()):
                    value = kwargs[key]
                    if isinstance(value, float):
                        model_name += '_{}{:3.2e}'.format(key, value)
                    else:
                        model_name += '_{}{}'.format(key, value)
                return model_name
            cut_settings = {
                 'mucut' : 5e-3,
                 'ccut' : 0.3,
                 'ethresh' : 500,
                 'angrescut' : float(np.deg2rad(30)),
                 'angfloor' : float(np.deg2rad(0.5))
            }
            model_dir = '{}/sigma_pdf_test/{}'.format(
                model_base_dir, get_model_name(**cut_settings))    

            observables = ['log10energy']
            if additionalpdfs == 'sigma_src':
                conditionals = ['srcsindec', 'sindec', 'sigma', 'gamma']
            else:    
                conditionals = ['sindec', 'sigma', 'gamma']
            observables = sorted(observables)
            obs_str = observables[0].replace('src->', 'src')
            for obs in observables[1:]:
                obs_str += '_' + obs.replace('src->', 'src')
                
            conditionals = sorted(conditionals)
            if len(conditionals) > 0:
                cond_str = conditionals[0].replace('src->', 'src')
            else:
                cond_str = 'None'
            for cond in conditionals[1:]:
                cond_str += '_' + cond.replace('src->', 'src')
            ratio_model_path = '{}/{}/{}/ratio_model_interp/{}'.format(
                     model_dir, obs_str, cond_str, ratio_model_name)


            ratio_model, info  = load_model(ratio_model_path)
            print('features:', ratio_model.features)
            print('parameters:', ratio_model.parameters)
            print('observables:', ratio_model.observables)
            print('conditionals:', ratio_model.conditionals) 
             
            #print(info)
            trained_cut_settings = info['info']['cut_settings']
            if trained_cut_settings['angrescut'] != np.deg2rad(30):
                print(trained_cut_settings['angrescut'])
                raise Exception('Angres cut doesnt Match!!')
            if trained_cut_settings['ccut'] != 0.3:
                print(trained_cut_settings['ccut'])
                raise Exception('CascadeBDT cut doesnt Match!!')
            if trained_cut_settings['mucut'] != 5e-3:
                print(trained_cut_settings['mucut'])
                raise Exception('MuonBDT cut doesnt Match!!')
            if trained_cut_settings['ethresh'] != 500:
                print(trained_cut_settings['ethresh'])
                raise Exception('Ethresh cut doesnt Match!!')
            if temp == 'pi0':
                template = repo.get_template ('Fermi-LAT_pi0_map')
                gp_conf = {
                    'template': template,
                    'flux':     cy.hyp.PowerLawFlux(2.5),
                    'fitter_args': dict(gamma=2.5),
                    'sigsub': True,
                    'fast_weight': True,
                    cy.pdf.GenericPDFRatioModel: dict(
                                func = ratio_model,
                                features = ratio_model.features,
                                fits = dict(gamma=np.r_[1, 1:4.01:.125, 4]),
                            ),
                    'update_bg': True,
                    'energy' : False #If using the NN energy needs to be set to False, 
                        }   
            elif 'kra' in temp:
                if temp =='kra5':
                    template, energy_bins = repo.get_template(
                              'KRA-gamma_5PeV_maps_energies', per_pixel_flux=True)
                elif temp =='kra50':
                    template, energy_bins = repo.get_template(
                              'KRA-gamma_50PeV_maps_energies', per_pixel_flux=True)
                gp_conf = {
                  # desired template
                  'template': template,
                  'bins_energy': energy_bins,
                  'fitter_args' : dict(gamma=2.5),
                  'update_bg' : True,
                  'sigsub': True,
                  'energy' : False
                  }
            if additionalpdfs == 'sigma':
                print('Space * E * Sigma')   
                gp_conf[cy.pdf.SigmaPDFRatioModel] = dict(
                            hkw=dict(bins=( np.linspace(-1,1,20),  
                            np.linspace(0, np.deg2rad(30), 20))),
                            features=['sindec', 'sigma'],
                            normalize_axes = ([1])) 
            tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus = cpus)
        else:
            if temp == 'pi0':
                template = repo.get_template ('Fermi-LAT_pi0_map')
                gp_conf = {
                    'template': template,
                    'flux':     cy.hyp.PowerLawFlux(2.5),
                    'fitter_args': dict(gamma=2.5),
                    'sigsub': True,
                    'fast_weight': True,
                    'dir': cy.utils.ensure_dir('{}/templates/pi0'.format(ana_dir))}
            elif temp =='kra5':
                template, energy_bins = repo.get_template(
                          'KRA-gamma_5PeV_maps_energies', per_pixel_flux=True)

                gp_conf = {
                    # desired template
                    'template': template,
                    'bins_energy': energy_bins,
                    'fitter_args' : dict(gamma=2.5),
                    'update_bg' : True,
                    'sigsub': True,
                    #'dir': cy.utils.ensure_dir('{}/templates/kra5'.format(ana_dir))
                    }
            if additionalpdfs == 'sigma':
                print('Space * E * Sigma')   
                gp_conf[cy.pdf.SigmaPDFRatioModel] = dict(
                            hkw=dict(bins=( np.linspace(-1,1,20),  np.linspace(0,
                            np.deg2rad(30), 20))),
                            features=['sindec', 'sigma'],
                            normalize_axes = ([1])) 
            tr = cy.get_trial_runner(gp_conf, ana=ana, mp_cpus = cpus)
        return tr
    tr = get_tr(nn, temp, additionalpdfs)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()

    bg = cy.dists.Chi2TSD(tr.get_many_fits (
      n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))

    template_sens = tr.find_n_sig(
                    bg.median(), 
                    0.9, #percent above threshold (0.9 for sens)
                    n_sig_step=10,
                    batch_size = n_trials / 3, 
                    tol = 0.02)

    if temp == 'pi0':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
    else:
        template_sens['model_norm'] = tr.to_model_norm(template_sens['n_sig'])

    flush ()

    if nn:
        out_dir = cy.utils.ensure_dir(
            '{}/NN{}/gp/{}/{}/'.format(
            state.base_dir, model_name, additionalpdfs, temp))
    else:
        out_dir = cy.utils.ensure_dir(
            '{}/gp/{}/{}/'.format(
            state.base_dir,temp , additionalpdfs))
    out_file = out_dir + 'sens.npy'
    print(template_sens)
    np.save(out_file, template_sens)
    print ('-> {}'.format (out_file))                                                          


@cli.command ()
@pass_state
def collect_gp_trials (state):
    bg = cy.bk.get_all (
        '{}/trials/{}/gp'.format (state.base_dir, state.ana_name),
        'trials*npy',
        merge=np.concatenate, post_convert=cy.utils.Arrays)
    outfile = '{}/trials/{}/gp.dict'.format (state.base_dir, state.ana_name)
    print ('->', outfile)
    with open (outfile, 'wb') as f:
        pickle.dump (bg, f, -1)

if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
