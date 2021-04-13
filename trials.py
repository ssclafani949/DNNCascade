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

global mucut
global ccut
global ethresh 

hostname = socket.gethostname()
print('Hostname: {}'.format(hostname))
if 'condor00' in hostname or 'cobol' in hostname or 'gpu' in hostname:
    print('Using UMD')
    repo = cy.selections.Repository(local_root='/data/i3store/users/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses/ECAS_11_yrs')
    job_basedir = '/data/i3home/ssclafani/submitter_logs'
else:
    repo = cy.selections.Repository(local_root='/data/user/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses/ECAS_11_yrs')
    ana_dir = '{}/ana'.format (base_dir)
    job_basedir = '/scratch/ssclafani/' 

class Cascades(cy.selections.MESEDataSpecs.MESCDataSpec):
    def __init__(self, mucut,ccut, angrescut, ethresh):
        self.mucut = mucut
        self.ccut = ccut
        self.angrescut = angrescut
        self.ethresh = ethresh 
    def dataset_modifications(self, ds):
         ds.data = ds.data._subsample(ds.data.mu_score <= self.mucut)
         ds.sig = ds.sig._subsample(ds.sig.mu_score <= self.mucut)
        
         ds.data = ds.data._subsample(ds.data.c_score >= self.ccut)
         ds.sig = ds.sig._subsample(ds.sig.c_score >= self.ccut)
        
         ds.data = ds.data._subsample(ds.data.energy > self.ethresh)
         ds.sig = ds.sig._subsample(ds.sig.energy > self.ethresh)
        
         ds.data = ds.data._subsample(ds.data.sigma <= np.deg2rad( self.angrescut))
         ds.sig = ds.sig._subsample(ds.sig.sigma <= np.deg2rad( self.angrescut ))
        
         d = ds.data
         n = ds.sig            
        
         true = astropy.coordinates.SkyCoord(ds.sig.true_ra, ds.sig.true_dec, unit='rad')
         reco = astropy.coordinates.SkyCoord(ds.sig.ra, ds.sig.dec, unit='rad')
         sep = true.separation(reco)
         dpsi = sep.rad
         #dpsi =  astro.angular_distance(n.true_ra, n.true_dec, n.ra, n.dec) 
         h = hl.hist((n.energy, dpsi / n.sigma), n.oneweight * n.true_energy**-2,
                      bins=(14,40), range=((10**2.5,1e7), (10**-2, 10**2)), log=True).normalize([1])
         hd = hl.hist((n.energy, dpsi / n.sigma), n.oneweight * n.true_energy**-2,
                      bins=(14,10**3), range=((10**2.5,1e7), (10**-2, 10**2)), log=True)
         h05 = hd.contain(1, .05)
         h95 = hd.contain(1, .95)                                                                     
         hmed = hd.median(1)
         s = hmed.spline_fit(s=0, log=True)
         n['uncorrected_sigma'] = np.copy(n.sigma)
         n['sigma'] = n.uncorrected_sigma * s(np.clip(n.energy, *hmed.range[0])) / 1.1774
         d['uncorrected_sigma'] = np.copy(d.sigma)
         d['sigma'] = d.uncorrected_sigma * s(np.clip(d.energy, *hmed.range[0])) / 1.1774

    # set livetime and data
    _keep = _keep = 'mjd true_energy oneweight'.split ()                       
    _keep32 = 'azimuth zenith ra dec energy sigma dist true_ra true_dec xdec xra mu_score c_score astro_bdt_01'.split () 
    _livetime = 3307.053 * 86400
    if 'condor00' in hostname or 'cobol' in hostname or 'gpu' in hostname: 
        _path_data = '/data/i3store/users/ssclafani/data/ECAS/2011_2021_loose_exp.npy'
        #BASELINE MC
        _path_sig = '/data/i3store/users/ssclafani/data/ECAS/IC86_2016_MC_loose_bfrv1.npy'
    else:
        _path_data = '/data/user/ssclafani/data/cscd/final/2011_2021_loose_exp.npy'
        #BASELINE MC
        _path_sig = '/data/user/ssclafani/data/cscd/final/IC86_2016_MC_loose_bfrv1.npy'


    _bins_sindec = np.linspace (-1, 1, 30+1)
    _bins_logenergy = np.arange (2, 8.5, .25)
    _kw_energy = dict(bins_sindec=np.linspace(-1,1, 31))

class State (object):
    def __init__ (self, ana_name, ana_dir, save,  base_dir, mucut, angrescut, ccut, ethresh, job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = ana_name, ana_dir, save, job_basedir
        self.base_dir = base_dir
        self.mucut = mucut
        self.ccut = ccut
        self.angrescut = angrescut
        self.ethresh = ethresh

        self._ana = None

    @property
    def ana (self):
        if self._ana is None:
            print(self.mucut, self.ccut, self.angrescut, self.ethresh)
            repo.clear_cache()
            specs = [Cascades(self.mucut,self.ccut, self.angrescut, self.ethresh)] #cy.selections.MESEDataSpecs.mesc_7yr + cy.selections.PSDataSpecs.ps_10yr
            ana = cy.analysis.Analysis (repo, specs)#r=self.ana_dir)
            #ana = cy.analysis.Analysis (repo, specs)
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
@click.option ('--mucut', default=1e-3)
@click.option ('--angrescut', default=20)
@click.option ('--ccut', default=0.3)
@click.option ('--ethresh', default=500)
@click.pass_context
def cli (ctx, ana_name, ana_dir, save, base_dir, mucut, angrescut, ccut, ethresh, job_basedir):
    ctx.obj = State.state = State (ana_name, ana_dir, save, base_dir, mucut, angrescut, ccut, ethresh, job_basedir)


@cli.resultcallback ()
def report_timing (result, **kw):
    exe_t1 = now ()
    print ('c7: end at {} .'.format (exe_t1))
    print ('c7: {} elapsed.'.format (exe_t1 - exe_t0))

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
@pass_state
def do_ps_sens ( state, n_trials, poisson,gamma, dec_deg, seed, cpus, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    ana = state.ana
    print('Doing Trials with: ')
    print('Muon BDT Cut : {}'.format(state.mucut))
    print('Cascade BDT Cut :{}'.format(state.ccut))
    print('Energy Threshold Cut :{}GeV'.format(state.ethresh))
    print('AngRes Cut: {}deg'.format( state.angrescut ))
    #dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir))
    sindec = np.sin(np.radians(dec_deg))
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

    t0 = now ()
    print ('Beginning calculation at {} ...'.format (t0))
    flush ()
    sens = get_PS_sens (sindec, gamma=gamma, n_trials=n_trials) 
    
    sens_flux = np.array(sens['flux'])
   
    out_dir = cy.utils.ensure_dir('{}/E{}_redo/mucut/{:.10f}/ccut/{}/angrescut/{}/ethresh/{}/dec/{:+08.3f}/'.format(
        state.base_dir, int(gamma*100), state.mucut, state.ccut, state.angrescut, state.ethresh, dec_deg))
    out_file = out_dir + 'sens.npy'
    print(sens_flux)
    np.save(out_file, sens_flux)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))


@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--sigmaPDF/--nosigmaPDF', default=False)
@click.option ('--gamma', default=2.0, type=float)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def get_bias_and_sens ( state, n_trials, poisson, sigmaPDF, gamma, dec_deg, seed, cpus, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    ana = state.ana
    print('Doing Trials with: ')
    print('Muon BDT Cut : {}'.format(state.mucut))
    print('Cascade BDT Cut :{}'.format(state.ccut))
    print('Energy Threshold Cut :{}GeV'.format(state.ethresh))
    print('AngRes Cut: {}deg'.format( state.angrescut ))
    #dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir))
    sindec = np.sin(np.radians(dec_deg))
    def get_PS_sens(sindec, n_trials=n_trials, gamma=gamma, mp_cpu=cpus, sigmaPDF=sigmaPFD):
        src = cy.utils.sources(0, np.arcsin(sindec), deg=False)
        if sigmaPDF:
            conf = {
                #'fitter_args' : {'gamma' : gamma}
                'ana' : ana,
                'src' : src,
                #'cut_n_sigma' : 5,
                #'sindec_bandwidth' : np.radians(5),
                'flux' : cy.hyp.PowerLawFlux(gamma),
                'update_bg' : False, 
                cy.pdf.SigmaPDFRatioModel : dict(
                    hkw=dict(bins=( np.linspace(-1,1,20),  np.linspace(0,np.pi, 12))), 
                    features=['sindec', 'sigma'],
                    normalize_axes = ([1])),     
            }
            tr = cy.get_trial_runner(conf=conf, mp_cpus=mp_cpu)
        else:
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

    t0 = now ()
    print ('Beginning calculation at {} ...'.format (t0))
    flush ()
    sens = get_PS_sens (sindec, gamma=gamma, n_trials=n_trials, sigmaPDF=sigmaPDF) 
    
    sens_flux = np.array(sens['flux'])
   
    out_dir = cy.utils.ensure_dir('{}/E{}/mucut/{:.10f}/ccut/{}/angrescut/{}/ethresh/{}/dec/{:+08.3f}/'.format(
        state.base_dir, int(gamma*100), state.mucut, state.ccut, state.angrescut, state.ethresh, dec_deg))
    out_file = out_dir + 'sens.npy'
    print(sens_flux)
    np.save(out_file, sens_flux)
    t1 = now ()
    print ('Finished trials at {} ...'.format (t1))

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--gamma', default=2, type=float)
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@pass_state
def submit_grid_search_sens (
        state, n_trials,  gamma, dry, seed):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = job_basedir 
    job_dir = '{}/{}/ECAS_11yr/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=8,  max_jobs=1000)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)


    mucuts = np.logspace(-4, -1, 31)
    ccuts = [0, 0.1, 0.3]
    ethreshes = [300,500, 700]
    angrescuts = [30, 80]
    for mucut in mucuts:
        for ccut in ccuts:
            for angrescut in angrescuts:
                for ethresh in ethreshes:
                    s =  seed
                    fmt = '{} --mucut {} --angrescut {} --ccut {} --ethresh {} do-all-sky-sens  --n-trials {}' \
                                        ' --gamma={:.3f}' \
                                        ' --seed={}'
                    command = fmt.format ( this_script, mucut, angrescut, ccut, ethresh,  n_trials,
                                          gamma, s)
                    #command = fmt.format (env_shell, this_script, mucut, ccut, ethresh,  n_trials,
                    #                      gamma, s)
                    fmt = 'csky__trials_{:07d}_mc_{:03f}_cc_{:03f}_arc_{:03f}_ethresh_{:03f}_' \
                            'gamma_{:.3f}_seed_{:04d}'
                    label = fmt.format (n_trials, mucut, ccut, angrescut, ethresh, gamma, s)
                    commands.append (command)
                    labels.append (label)
    #mucuts = np.logspace(-4, -1, 31)
    #ccuts = [0, 0.1, 0.3]
    #ethreshes = [300,500, 700]
    #angrescuts = [30, 80]
    sub.dry = dry
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--gamma', default=2, type=float)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@pass_state
def submit_do_ps_sens (
        state, n_trials,  gamma,dec_deg,  dry, seed):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/ECAS_11yr/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=8,  max_jobs=1000)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)


    #mucuts = [0.001]
    #ccuts = [0.3]
    #ethreshes = [500]
    #angrescuts = [30]
    #dec_degs = [-30]
    

    mucuts = np.logspace(-4, -1, 21)
    #mucuts = np.logspace(-3.1, -1.9, 9)
    ccuts = [0.0,  0.1, 0.3, 0.9]
    ethreshes = [500, 1000]
    angrescuts = [30]
    dec_degs = np.arange(-88, 89, 4)  
    for mucut in mucuts:
        for ccut in ccuts:
            for angrescut in angrescuts:
                for ethresh in ethreshes:
                    for dec_deg in dec_degs:
                        s =  seed
                        fmt = '{} --mucut {} --angrescut {} --ccut {} --ethresh {} do-ps-sens  --n-trials {}' \
                                            ' --gamma={:.3f} --dec_deg {}' \
                                            ' --seed={}'
                        command = fmt.format ( this_script, mucut, angrescut, ccut, ethresh,  n_trials,
                                              gamma, dec_deg, s)
                        fmt = 'csky_sens_{:07d}_mc_{:03f}_cc_{:03f}_arc_{:03f}_ethresh_{:03f}_' \
                                'gamma_{:.3f}_decdeg_{:04d}_seed_{:04d}'
                        label = fmt.format (n_trials, mucut, ccut, angrescut, ethresh, gamma, dec_deg, s)
                        commands.append (command)
                        labels.append (label)
    sub.dry = dry
    print(hostname)
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--gamma', default=2.0, type=float)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def do_ps_trials ( state, dec_deg, n_trials, gamma, cutoff, n_sig, poisson, seed, cpus, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    print('Doing Trials with: ')
    print('Muon BDT Cut : {}'.format(state.mucut))
    print('Cascade BDT Cut :{}'.format(state.ccut))
    print('Energy Threshold Cut :{}GeV'.format(state.ethresh))
    print('AngRes Cut: {}deg'.format( state.angrescut ))
    dec = np.radians(dec_deg)
    ana = state.ana
    cutoff_GeV = cutoff * 1e3
    dir = cy.utils.ensure_dir ('{}/ps/'.format (state.base_dir, dec_deg))
    tr = cy.get_trial_runner(src = cy.utils.Sources(dec=dec, ra=0) , ana=ana,
        flux=cy.hyp.PowerLawFlux(gamma, energy_cutoff = cutoff_GeV), mp_cpus=cpus, dir=dir)
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
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--n-jobs', default=10, type=int)
@click.option ('-n', '--n-sig', 'n_sigs', multiple=True, default=[0], type=float)
@click.option ('--gamma', default=2, type=float)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dec', 'dec_degs', multiple=True, type=float, default=())
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@pass_state
def submit_do_ps_trials (
        state, n_trials, n_jobs, n_sigs, gamma, cutoff,  poisson, dec_degs, dry, seed):
    ana_name = state.ana_name
    T = time.time ()
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_basedir = state.job_basedir 
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_dir = '{}/{}/ps_trials/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=8, max_jobs=1000)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    dec_degs = dec_degs or np.r_[-89:+89.01:2]
    mucut = 1e-3
    angrescut = 20
    ethresh = 500
    ccut = 0.3
    for dec_deg in dec_degs:
        for n_sig in n_sigs:
            for i in range (n_jobs):
                s = i + seed
                fmt = ' {} --mucut {} --angrescut {} --ethresh {} --ccut {} do-ps-trials --dec_deg={:+08.3f} --n-trials={}' \
                        ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                        ' --{} --seed={}'
  
                command = fmt.format (this_script, mucut,  angrescut, ethresh, dec_deg, n_trials,
                                      n_sig, gamma, cutoff, poisson_str, s)
                fmt = 'csky__dec_{:+08.3f}__trials_{:07d}__n_sig_{:08.3f}__' \
                        'gamma_{:.3f}_cutoff_{}_{}__seed_{:04d}'
                label = fmt.format (dec_deg, n_trials, n_sig, gamma,
                                    cutoff, poisson_str, s)
                commands.append (command)
                labels.append (label)
    sub.dry = dry
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


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
@pass_state
def do_gp_trials ( state, temp, n_trials, n_sig, poisson, seed, cpus, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
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
        kra5_map, kra5_energy_bins = repo.get_template(
                    'KRA-gamma_5PeV_maps_energies', per_pixel_flux=True)
        gp_conf = {
            # desired template
            'template': template,
            'bins_energy': kra5_energy_bins,
            'fitter_args': dict(gamma=2.5),
            'update_bg' : True,
            'sigsub': True,
            'dir': cy.utils.ensure_dir('{}/templates/kra5'.format(ana_dir))}

    tr = cy.get_trial_runner(gp_conf, ana=ana)
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
@click.option ('--seed', default=None, type=int)
@click.option ('--cpus', default=1, type=int)
@pass_state
def do_gp_sens ( state, temp, n_trials, seed, cpus, logging=True):
    if seed is None:
        seed = int (time.time () % 2**32)
    random = cy.utils.get_random (seed) 
    print(seed)
    ana = state.ana
    dir = cy.utils.ensure_dir ('{}/templates/{}'.format (state.base_dir, temp))
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

    tr = cy.get_trial_runner(gp_conf, ana=ana)
    t0 = now ()
    print ('Beginning trials at {} ...'.format (t0))
    flush ()

    bg = cy.dists.Chi2TSD(tr.get_many_fits (
        n_trials, n_sig=0, poisson=False, seed=seed, logging=logging))
    t1 = now ()
    print ('Finished bg trials at {} ...'.format (t1))

    template_sens = tr.find_n_sig(bg.median(), 0.9, n_sig_step=10,
        batch_size = n_trials / 3, tol = 0.02, mp_cups=cpus)
   
    if temp == 'pi0':
        template_sens['fluxE2_100TeV'] = tr.to_E2dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
        template_sens['flux_100TeV'] = tr.to_dNdE(template_sens['n_sig'], 
            E0 = 100 , unit = 1e3)
    else:
        template_sens['model_norm'] = tr.to_model_norm(template_sens['n_sig'])

    flush ()

    out_dir = cy.utils.ensure_dir('{}/gp/{}/mucut/{}/ccut/{}/angrescut/{}/ethresh/{}/'.format(
        state.base_dir,temp , state.mucut, state.ccut, state.angrescut,
        state.ethresh))
    out_file = out_dir + 'sens.npy'
    print(template_sens)
    np.save(out_file, template_sens)
    print ('-> {}'.format (out_file))                                                          

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option ('--template', default='kra5')
@pass_state
def submit_gp_sens (
        state, n_trials, dry, seed, template):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir #'/scratch/ssclafani/' 
    job_dir = '{}/{}/ECAS_gp/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=8,  max_jobs=1000)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    
    mucuts = np.logspace(-4, -1, 21)
    ccuts = [0.3]
    ethreshes = [300,500,1000]
    angrescuts = [30, 80]
    for mucut in mucuts:
        for ccut in ccuts:
            for angrescut in angrescuts:
                for ethresh in ethreshes:
                    s =  seed
                    fmt = '{} --mucut {} --angrescut {} --ccut {} --ethresh {} do-gp-sens  --n-trials {}' \
                                        ' --seed={} {}'                                                    
                    command = fmt.format ( this_script, mucut, angrescut, ccut, ethresh,  n_trials,
                                         s, template)
                    fmt = 'csky__trials_{:07d}_mc_{:03f}_ccut_{:03f}_arc_{:03f}_ethresh_{:03f}_' \
                            'gp_{}_seed_{:04d}'
                    label = fmt.format (n_trials, mucut, ccut, angrescut, ethresh, template,  s)
                    commands.append (command)
                    labels.append (label)
    sub.dry = dry
    print(hostname)
    if 'condor00' in hostname:
        print('submitting from condor00')
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


@cli.command ()
@click.argument ('temp')
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--n-jobs', default=10, type=int)
@click.option ('-n', '--n-sig', 'n_sigs', multiple=True, default=[0], type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dry/--nodry', default=False)
@click.option ('-b', '--blacklist', multiple=True)
@click.option ('--seed', default=0, type=int)
@pass_state
def submit_do_gp_trials (
        state, temp, n_trials, n_jobs, n_sigs, poisson, dry, blacklist, seed):
    
    #example command using click python comb_sens.py submit_do_gp_trials --n-sig=0 --n-jobs=1 --n-trials=1000 pi0
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_dir = '{}/{}/gp_trials/{}/T_{:17.6f}'.format (
        job_basedir, ana_name, temp, T)
    sub = Submitter (job_dir=job_dir, memory=10)#, max_jobs=400)
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    for n_sig in n_sigs:
        for i in xrange (n_jobs):
            s = i + seed
            fmt = '{} {} {} do_gp_trials --n-trials={}' \
                    ' --n-sig={}' \
                    ' --{} --seed={} {}'
            command = fmt.format (env_shell, this_script, state.state_args, n_trials,
                                  n_sig, poisson_str,  s, temp)
            fmt = 'csky__trials_{:07d}__n_sig_{:08.3f}__' \
                    '{}__{}__seed_{:04d}'
            label = fmt.format (n_trials,  n_sig, temp, poisson_str, s)
            commands.append (command)
            labels.append (label)
    #print(commands)
    #sub.dry = dry
    if 'condor00' in hostname:
        print('submitting from condor00')
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)

    sub.submit_npx4 (
        commands, labels)
        #blacklist=cg.blacklist (blacklist))

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
