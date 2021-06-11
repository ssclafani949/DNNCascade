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
    base_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses/DNNC/')
    job_basedir = '/data/i3home/ssclafani/submitter_logs'
else:
    repo = cy.selections.Repository(local_root='/data/user/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses/DNNC')
    ana_dir = '{}/ana'.format (base_dir)
    job_basedir = '/scratch/ssclafani/' 

class State (object):
    def __init__ (self, ana_name, ana_dir, save,  base_dir,  job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = ana_name, ana_dir, save, job_basedir
        self.base_dir = base_dir
        self._ana = None

    @property
    def ana (self):
        if self._ana is None:
            specs = cy.selections.DNNCasacdeDataSpecs.DNNC_11yr
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
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--nn/--nonn', default=True)
@click.option ('--seed', default=0)
@pass_state
def submit_do_ps_trials (
        state, n_trials, n_jobs, n_sigs, gamma, cutoff,  poisson, dec_degs, dry, 
        model_name, additionalpdfs, nn, seed):
    ana_name = state.ana_name
    T = time.time ()
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_basedir = state.job_basedir 
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_dir = '{}/{}/ps_trials/T_E{}_{:17.6f}'.format (
        job_basedir, ana_name, int(gamma * 100),  T)
    sub = Submitter (job_dir=job_dir, memory=8, max_jobs=1000)
    commands, labels = [], []
    trial_script = os.path.abspath('trials.py')
    dec_degs = dec_degs or np.r_[-89:+89.01:2]
    for dec_deg in dec_degs:
        for n_sig in n_sigs:
            for i in range (n_jobs):
                s = i + seed
                if nn:
                    fmt = ' {} do-ps-trials --dec_deg={:+08.3f} --n-trials={}' \
                            ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                            ' --{} --seed={} --model_name {} --nn '
      
                    command = fmt.format (trial_script,  dec_deg, n_trials,
                                          n_sig, gamma, cutoff, poisson_str, s, model_name)
                    fmt = 'csky__dec_{:+08.3f}__trials_{:07d}__n_sig_{:08.3f}__' \
                            'gamma_{:.3f}_cutoff_{}_{}__seed_{:04d}_nn_{}'
                    label = fmt.format (dec_deg, n_trials, n_sig, gamma,
                                        cutoff, poisson_str, s, model_name)
                else:
                    fmt = ' {} do-ps-trials --dec_deg={:+08.3f} --n-trials={}' \
                            ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                            ' --{} --seed={} --nonn '
      
                    command = fmt.format (trial_script,  dec_deg, n_trials,
                                          n_sig, gamma, cutoff, poisson_str, s,)
                    fmt = 'csky__dec_{:+08.3f}__trials_{:07d}__n_sig_{:08.3f}__' \
                            'gamma_{:.3f}_cutoff_{}_{}__seed_{:04d}'
                    label = fmt.format (dec_deg, n_trials, n_sig, gamma,
                                        cutoff, poisson_str, s )

                commands.append (command)
                labels.append (label)
    sub.dry = dry
    print(hostname)
    if nn:
        if 'condor00' in hostname:
            sub.submit_condor00 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
        else:
            sub.submit_npx4 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
    else:
        if 'condor00' in hostname:
            sub.submit_condor00 (commands, labels)
        else:
            sub.submit_npx4 (commands, labels)

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--gamma', default=2, type=float)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--dry/--nodry', default=False)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--seed', default=0)
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--nn/--nonn', default=True)
@pass_state                                                                                                               
def submit_do_ps_sens (
        state, n_trials,  gamma,dec_deg,  dry, additionalpdfs, seed, model_name, nn):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/ECAS_11yr/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=8,  max_jobs=1000)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    trial_script = os.path.abspath('trials.py')
    sindecs = np.arange(-1,1.01,.1)
    sindecs[0] = -.99
    sindecs[-1] = .99
    dec_degs = np.degrees(np.arcsin(sindecs))
    additionalpdfs = ['None', 'sigma']
    model_names = ['small_scipy1d']    
    for dec_deg in dec_degs:
        for addpdf in additionalpdfs:                                                                                
            for model_name in model_names:
                s =  seed
                if nn:
                    fmt = '{} do-ps-sens  --n-trials {}' \
                                        ' --gamma={:.3f} --dec_deg {} --model_name {}'  \
                                        ' --seed={} --additionalpdfs={} --nn '
                    command = fmt.format ( trial_script,  n_trials,
                                          gamma, dec_deg, model_name, s, addpdf)
                    fmt = 'csky_sens_{:07d}_' \
                            'gamma_{:.3f}_decdeg_{:04f}_seed_{:04d}_{}nn_{}'
                    label = fmt.format (
                        n_trials, mucut, ccut, angrescut, 
                        ethresh, gamma, dec_deg, s, addpdf, 
                        model_name)
                else:
                    fmt = '{} do-ps-sens  --n-trials {}' \
                                        ' --gamma={:.3f} --dec_deg {}' \
                                        ' --seed={} --additionalpdfs={} --nonn'
                    command = fmt.format ( trial_script,  n_trials,
                                          gamma, dec_deg, s, addpdf)
                    fmt = 'csky_sens_{:07d}_' \
                            'gamma_{:.3f}_decdeg_{:04f}_seed_{:04d}_{}'
                    label = fmt.format (
                            n_trials, 
                            gamma, dec_deg, s,
                            addpdf)
                commands.append (command)
                labels.append (label)
    sub.dry = dry
    print(hostname)
    if nn:
        if 'condor00' in hostname:
            sub.submit_condor00 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
        else:
            sub.submit_npx4 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
    else:
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
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--nn/--nonn', default=True)
@click.option ('--dry/--nodry', default=False)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--seed', default=0, type=int)
@pass_state
def submit_do_gp_trials (
        state, temp, n_trials, n_jobs, n_sigs, 
        poisson, model_name, additionalpdfs, nn, dry, cutoff, seed):
    #example command using click python submit.py submit-do-gp-trials --n-sig=0 --n-jobs=1 --n-trials=1000 pi0
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_dir = '{}/{}/gp_trials/{}/T_{:17.6f}'.format (
        job_basedir, ana_name, temp, T)
    sub = Submitter (job_dir=job_dir, memory=8)#, max_jobs=400)
    commands, labels = [], []
    trial_script = os.path.abspath('trials.py')
    print(n_sigs)
    for n_sig in n_sigs:
        for i in range (n_jobs):
            if nn:
                s = i + seed
                fmt = '{} do-gp-trials --n-trials={}' \
                        ' --n-sig={} --additionalpdfs {} --model_name {}' \
                        ' --{} --seed={} --nn --cutoff {} {}'
                command = fmt.format (trial_script,  n_trials,
                                      n_sig, additionalpdfs, model_name, poisson_str,  s, temp)
                fmt = 'csky__trials_{:07d}__n_sig_{:08.3f}__' \
                        '{}__{}__seed_{:04d}_nn_{}_{}_cutoff_{}'
                label = fmt.format (
                        n_trials,  n_sig, temp, poisson_str, 
                        s, model_name, additionalpdfs, cutoff)
            else:
                s = i + seed
                fmt = '{} do-gp-trials --n-trials={}' \
                        ' --n-sig={} --additionalpdfs {}' \
                        ' --{} --seed={} --cutoff {} --nonn {}'
                command = fmt.format (trial_script,  n_trials,
                                      n_sig, additionalpdfs, poisson_str,  s, cutoff, temp)
                fmt = 'csky__trials_{:07d}__n_sig_{:08.3f}__' \
                        '{}__{}__seed_{:04d}__cutoff_{}'
                label = fmt.format (
                        n_trials,  n_sig, temp, poisson_str, 
                        s,   cutoff)

            commands.append (command)
            labels.append (label)
    if 'condor00' in hostname:
        print('submitting from condor00')
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option ('--template', default='kra5')
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--nn/--nonn', default=True)
@pass_state
def submit_gp_sens (
        state, n_trials, dry, seed, template, model_name, nn):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir #'/scratch/ssclafani/' 
    job_dir = '{}/{}/ECAS_gp/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=8,  max_jobs=1000)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    trial_script = os.path.abspath('trials.py')
    additionalpdfs = ['None', 'sigma']
    for addpdf in additionalpdfs:
        if nn:
            s =  seed
            fmt = '{} do-gp-sens  --n-trials {}' \
                                ' --seed={} --model_name {} --additionalpdfs {} --nn {}'
            command = fmt.format ( trial_script, n_trials,
                                 s, model_name, addpdf, template)
            fmt = 'csky__trials_{:07d}_' \
                    'gp_{}_seed_{:04d}_nn_{}_{}'
            label = fmt.format (n_trials, template,  s, model_name, addpdf)
            commands.append (command)
            labels.append (label)
        else:
            s =  seed
            fmt = '{} do-gp-sens  --n-trials {}' \
                                ' --seed={} --nonn  --additionalpdfs {} {}'
            command = fmt.format ( trial_script, n_trials,
                                 s, addpdf, template)
            fmt = 'csky__trials_{:07d}_' \
                    'gp_{}_seed_{:04d}_{}'
            label = fmt.format (n_trials, template,  s, addpdf)
            commands.append (command)
            labels.append (label)
    sub.dry = dry
    print(hostname)
    if nn:
        if 'condor00' in hostname:
            sub.submit_condor00 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
        else:
            sub.submit_npx4 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
    else:
        if 'condor00' in hostname:
            print('submitting from condor00')
            sub.submit_condor00 (commands, labels)
        else:
            sub.submit_npx4 (commands, labels)

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--n-jobs', default=10, type=int)
@click.option ('-n', '--n-sig', 'n_sigs', multiple=True, default=[0], type=float)
@click.option ('--gamma', default=2, type=float)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dry/--nodry', default=False)
@click.option ('--model_name', default='small_scipy1d', type=str)
@click.option ('--catalog', type=str, default=None)
@click.option ('--additionalpdfs', type=str, default=None)
@click.option ('--nn/--nonn', default=True)
@click.option ('--seed', default=0)
@pass_state
def submit_do_stacking_trials (
        state, n_trials, n_jobs, n_sigs, gamma, cutoff,  poisson,  dry, 
        model_name, catalog, additionalpdfs, nn, seed):
    ana_name = state.ana_name
    T = time.time ()
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/stacking_trials/T_E{}_{:17.6f}'.format (
        job_basedir, ana_name, int(gamma * 100),  T)
    sub = Submitter (job_dir=job_dir, memory=8, max_jobs=1000)
    commands, labels = [], []
    trial_script = os.path.abspath('trials.py')
    if catalog:
        catalogs = [catalog]
    else:
        catalogs = ['snr', 'unid', 'pwn']
    for cat in catalogs:
        for n_sig in n_sigs:
            for i in range (n_jobs):
                s = i + seed
                if nn:
                    fmt = ' {} do-stacking-trials --catalog={} --n-trials={}' \
                            ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                            ' --{} --seed={} --model_name {} --nn '
      
                    command = fmt.format (trial_script,  cat, n_trials,
                                          n_sig, gamma, cutoff, poisson_str, s, model_name)
                    fmt = 'csky__cat_{}__trials_{:07d}__n_sig_{:08.3f}__' \
                            'gamma_{:.3f}_cutoff_{}_{}__seed_{:04d}_nn_{}'
                    label = fmt.format (cat, n_trials, n_sig, gamma,
                                        cutoff, poisson_str, s, model_name)
                else:
                    fmt = ' {} do-stacking-trials --catalog={} --n-trials={}' \
                            ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                            ' --{} --seed={} --nonn '
      
                    command = fmt.format (trial_script,  cat, n_trials,
                                          n_sig, gamma, cutoff, poisson_str, s)
                    fmt = 'csky__cat_{}__trials_{:07d}__n_sig_{:08.3f}__' \
                            'gamma_{:.3f}_cutoff_{}_{}__seed_{:04d}'
                    label = fmt.format (cat, n_trials, n_sig, gamma,
                                        cutoff, poisson_str, s)

                commands.append (command)
                labels.append (label)
    sub.dry = dry
    print(hostname)
    if nn:
        if 'condor00' in hostname:
            sub.submit_condor00 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
        else:
            sub.submit_npx4 (commands, labels, reqs = '(TARGET.has_avx2) && (TARGET.has_ssse3)')
    else:
        if 'condor00' in hostname:
            sub.submit_condor00 (commands, labels)
        else:
            sub.submit_npx4 (commands, labels)



if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
