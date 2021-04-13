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
    trial_script = os.path.abspath('trials.py')
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
  
                command = fmt.format (trial_script, mucut,  angrescut, ethresh, dec_deg, n_trials,
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
    trial_script = os.path.abspath('trials.py')
    env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    for n_sig in n_sigs:
        for i in xrange (n_jobs):
            s = i + seed
            fmt = '{} {} {} do_gp_trials --n-trials={}' \
                    ' --n-sig={}' \
                    ' --{} --seed={} {}'
            command = fmt.format (trial_script, state.state_args, n_trials,
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

if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
