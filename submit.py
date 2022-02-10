#!/usr/bin/env python
try:
    import matplotlib
except ModuleNotFoundError:
    os.environ['MPLBACKEND'] = 'AGG'
    import matplotlib

import csky as cy
import numpy as np
import datetime, socket
from submitter import Submitter
now = datetime.datetime.now
import config as cg
import click, sys, os, time
flush = sys.stdout.flush

repo, ana_dir, base_dir, job_basedir = cg.repo, cg.ana_dir, cg.base_dir, cg.job_basedir
hostname = cg.hostname
username = cg.username
submit_cfg_file = cg.submit_cfg_file


class State (object):
    def __init__ (self, ana_name, ana_dir, save,  base_dir,  job_basedir):
        self.ana_name, self.ana_dir, self.save, self.job_basedir = ana_name, ana_dir, save, job_basedir
        self.base_dir = base_dir
        self._ana = None    
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
@click.option ('--sigsub/--nosigsub', default=True)
@click.option ('--dec_degs', 'dec_degs', multiple=True, type=float, default=())
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option('--memory', default=0.5, type=float, help='Requested memory (GB)')
@pass_state
def submit_do_ps_trials (
        state, n_trials, n_jobs, n_sigs, gamma, 
        cutoff,  poisson, sigsub, dec_degs, dry, 
        seed, memory):
    ana_name = state.ana_name
    T = time.time ()
    poisson_str = 'poisson' if poisson else 'nopoisson'
    sigsub_str = 'sigsub' if sigsub else 'nosigsub'
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/ps_trials/T_E{}_{:17.6f}'.format (
        job_basedir, ana_name, int(gamma * 100),  T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
    commands, labels = [], []
    #reqs = '(Machine != "cobol97.private.pa.umd.edu") & (Machine != "cobol94.private.pa.umd.edu")'
    trial_script = os.path.abspath('trials.py')
    dec_degs = dec_degs or np.r_[-81:+81.01:2]
    for dec_deg in dec_degs:
        for n_sig in n_sigs:
            for i in range (n_jobs):
                s = i + seed
                fmt = ' {} do-ps-trials --dec_deg={:+08.3f} --n-trials={}' \
                        ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                        ' --{} --seed={} --{}'
  
                command = fmt.format (trial_script,  dec_deg, n_trials,
                                      n_sig, gamma, cutoff, poisson_str, s,sigsub_str)
                fmt = 'csky__dec_{:+08.3f}__trials_{:07d}__n_sig_{:08.3f}__' \
                        'gamma_{:.3f}_cutoff_{}_{}__seed_{:04d}'
                label = fmt.format (dec_deg, n_trials, n_sig, gamma,
                                    cutoff, poisson_str, s )
                commands.append (command)
                labels.append (label)
    sub.dry = dry
    print(hostname)
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels) #, reqs=reqs)
    else:
        sub.submit_npx4 (commands, labels)

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--gamma', default=2, type=float)
@click.option ('--dec_deg',   default=0, type=float, help='Declination in deg')
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option('--memory', default=5, type=float, help='Requested memory (GB)')
@pass_state                                                                                                               
def submit_do_ps_sens (
        state, n_trials,  gamma,dec_deg,  dry, seed, memory):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/ECAS_11yr/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
    #env_shell = os.getenv ('I3_BUILD') + '/env-shell.sh'
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    trial_script = os.path.abspath('trials.py')
    sindecs = np.arange(-1,1.01,.1)
    sindecs[0] = -.99
    sindecs[-1] = .99
    dec_degs = np.degrees(np.arcsin(sindecs))
    for dec_deg in dec_degs:
        s =  seed
        fmt = '{} do-ps-sens  --n-trials {}' \
                            ' --gamma={:.3f} --dec_deg {}' \
                            ' --seed={}'
        command = fmt.format ( trial_script,  n_trials,
                              gamma, dec_deg, s)
        fmt = 'csky_sens_{:07d}_' \
                'gamma_{:.3f}_decdeg_{:04f}_seed_{:04d}'
        label = fmt.format (
                n_trials, 
                gamma, dec_deg, s)
        commands.append (command)
        labels.append (label)
    sub.dry = dry
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


@cli.command()
@click.option('--n-trials', default=1000, type=int)
@click.option('--n-jobs', default=10, type=int)
@click.option('--dry/--nodry', default=False)
@click.option('--seed', default=0)
@click.option('--memory', default=3.5, type=float, help='Requested memory (GB)')
@pass_state
def submit_do_correlated_trials_sourcelist(
        state, n_trials, n_jobs,  dry, seed, memory):
    ana_name = state.ana_name
    T = time.time()
    job_basedir = state.job_basedir
    job_dir = '{}/{}/correlated_trials_sourcelist/T_{:17.6f}'.format(
        job_basedir, ana_name,  T)
    sub = Submitter(
        job_dir=job_dir, memory=memory,
        max_jobs=1000, config=submit_cfg_file)

    commands, labels = [], []
    trial_script = os.path.abspath('trials.py')
    for i in range(n_jobs):
        s = i + seed
        fmt = '{} do-correlated-trials-sourcelist --n-trials {} --seed={}'
        command = fmt.format(trial_script,  n_trials, s)
        fmt = 'csky_correlated_trials_sourcelist_{:07d}_seed_{:04d}'
        label = fmt.format(n_trials, s)
        commands.append(command)
        labels.append(label)
    sub.dry = dry
    if 'condor00' in hostname:
        sub.submit_condor00(commands, labels)
    else:
        sub.submit_npx4(commands, labels)


@cli.command()
@click.option('--n-trials', default=50000, type=int)
@click.option('--n-jobs', default=10, type=int)
@click.option('--dry/--nodry', default=False)
@click.option('--seed', default=0)
@click.option('--memory', default=4.5, type=float, help='Requested memory (GB)')
@pass_state
def submit_do_correlated_trials_fermibubbles(
        state, n_trials, n_jobs,  dry, seed, memory):
    ana_name = state.ana_name
    T = time.time()
    job_basedir = state.job_basedir
    job_dir = '{}/{}/correlated_trials_fermibubbles/T_{:17.6f}'.format(
        job_basedir, ana_name,  T)
    sub = Submitter(
        job_dir=job_dir, memory=memory,
        max_jobs=1000, config=submit_cfg_file)

    commands, labels = [], []
    trial_script = os.path.abspath('trials.py')
    for i in range(n_jobs):
        s = i + seed
        fmt = '{} do-correlated-trials-fermibubbles --n-trials {} --seed={}'
        command = fmt.format(trial_script,  n_trials, s)
        fmt = 'csky_correlated_trials_fermibubbles_{:07d}_seed_{:04d}'
        label = fmt.format(n_trials, s)
        commands.append(command)
        labels.append(label)
    sub.dry = dry
    if 'condor00' in hostname:
        sub.submit_condor00(commands, labels)
    else:
        sub.submit_npx4(commands, labels)


@cli.command ()
@click.option ('--n-jobs', default=10, type=int)
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--gamma', default=2, type=float)
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option ('-sourcenum', multiple=True, default=None, type=int)
@click.option('--memory', default=0.5, type=float, help='Requested memory (GB)')
@pass_state                                                                                                               
def submit_do_bkg_trials_sourcelist (
        state, n_jobs, n_trials,  gamma,  dry, seed, sourcenum, memory):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/correlated_trials_sourcelist_bkg/T_{:17.6f}'.format(
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    trial_script = os.path.abspath('trials.py')
    if sourcenum:
        sources = sourcenum
    else:
        nsources = 109
        sources = [int(source) for source in range(nsources)]
    print('Submitting For Sources:')
    print(sources)   
    for source in sources:
        for i in range (n_jobs):
            s = i + seed
            fmt = '{} do-bkg-trials-sourcelist  --n-trials {}' \
                                ' --sourcenum {}' \
                                ' --seed={}'
            command = fmt.format ( trial_script,  n_trials,
                                   source, s)
            fmt = 'csky_sens_{:07d}_' \
                    'source_{}_seed_{:04d}'
            label = fmt.format (
                    n_trials, 
                    source, s)
            commands.append (command)
            labels.append (label)
    sub.dry = dry
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


@cli.command ()
@click.argument ('temp')
@click.option ('--n-trials', default=50000, type=int)
@click.option ('--n-jobs', default=10, type=int)
@click.option ('-n', '--n-sig', 'n_sigs', multiple=True, default=[0], type=float)
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dry/--nodry', default=False)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--seed', default=0, type=int)
@click.option('--memory', default=2, type=float, help='Requested memory (GB)')
@pass_state
def submit_do_gp_trials (
        state, temp, n_trials, n_jobs, n_sigs, 
        poisson, dry, cutoff, seed, memory):
    #example command using click python submit.py submit-do-gp-trials --n-sig=0 --n-jobs=1 --n-trials=1000 pi0
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_dir = '{}/{}/gp_trials/{}/T_{:17.6f}'.format (
        job_basedir, ana_name, temp, T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
    commands, labels = [], []
    reqs = '(Machine != "cobol93.private.pa.umd.edu")'
    trial_script = os.path.abspath('trials.py')
    print(n_sigs)
    for n_sig in n_sigs:
        for i in range (n_jobs):
            s = i + seed
            fmt = '{} do-gp-trials --n-trials={}' \
                    ' --n-sig={} ' \
                    ' --{} --seed={} --cutoff {} {}'
            command = fmt.format (trial_script,  n_trials,
                                  n_sig,  poisson_str,  s, cutoff, temp)
            fmt = 'csky__trials_{:07d}__n_sig_{:08.3f}__' \
                    '{}__{}__seed_{:04d}__cutoff_{}'
            label = fmt.format (
                    n_trials,  n_sig, temp, poisson_str, 
                    s,   cutoff)
            commands.append (command)
            labels.append (label)
    if 'condor00' in hostname:
        print('submitting from condor00')
        sub.submit_condor00 (commands, labels, reqs=reqs)
    else:
        sub.submit_npx4 (commands, labels)

@cli.command ()
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--dry/--nodry', default=False)
@click.option ('--gamma', default=2.5)
@click.option ('--cutoff', default=np.inf)
@click.option ('--nsigma', default=None)
@click.option ('--seed', default=0)
@click.option ('--template', default='kra5')
@click.option('--memory', default=5, type=float, help='Requested memory (GB)')
@pass_state
def submit_gp_sens (
        state, n_trials, dry, gamma, cutoff, seed, template, nsigma, memory):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir #'/scratch/ssclafani/' 
    job_dir = '{}/{}/ECAS_gp/T_{:17.6f}'.format (
        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    trial_script = os.path.abspath('trials.py')
    s =  seed
    if nsigma:
        fmt = '{} do-gp-sens  --n-trials {}' \
                            ' --seed={} --gamma {} --nsigma {} --cutoff {} {}'
        command = fmt.format ( trial_script, n_trials,
                             s, gamma, nsigma, cutoff, template)
        fmt = 'csky__trials_{:07d}_' \
                'gp_{}_gamma_{}_cutoff_{}_seed_{:04d}_{}sigma'
        label = fmt.format (n_trials, template,  gamma, cutoff, s, nsigma)
    else:
        fmt = '{} do-gp-sens  --n-trials {}' \
                            ' --seed={} --gamma {} --cutoff {} {}'
        command = fmt.format ( trial_script, n_trials,
                             s, gamma, cutoff, template)
        fmt = 'csky__trials_{:07d}_' \
                'gp_{}_gamma_{}_cutoff_{}_seed_{:04d}'
        label = fmt.format (n_trials, template,  gamma, cutoff, s)
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
@click.option ('--n-trials', default=10000, type=int)
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option ('--template', default='kra5')
@click.option('--memory', default=5, type=float, help='Requested memory (GB)')
@pass_state
def submit_gp_erange (
        state, n_trials, dry, seed, template, memory):
    ana_name = state.ana_name
    T = time.time ()
    job_basedir = state.job_basedir #'/scratch/ssclafani/' 
    job_dir = '{}/{}/gp_erange/T_{:17.6f}'.format (

        job_basedir, ana_name,  T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
    commands, labels = [], []
    this_script = os.path.abspath (__file__)
    trial_script = os.path.abspath('trials.py')
    emins = np.round(np.linspace(500,10000, 21 ), 2)
    emaxs = np.round(np.logspace(4.6,8,21), 2)
    for emin in emins:
        emax = 1e8
        s =  seed
        fmt = '{} do-gp-sens-erange  --n-trials {}' \
                            ' --seed={} --emin {} --emax {} {}'
        command = fmt.format ( trial_script, n_trials,
                             s, emin, emax, template)
        fmt = 'csky__trials_{:07d}_' \
                'gp_{}_seed_{:04d}_emin{}_emax_{}'
        label = fmt.format (n_trials, template,  s, emin, emax)
        commands.append (command)
        labels.append (label)
    for emax in emaxs:
        emin = 500 
        s =  seed
        fmt = '{} do-gp-sens-erange  --n-trials {}' \
                            ' --seed={} --emin {} --emax {}  {}'
        command = fmt.format ( trial_script, n_trials,
                             s, emin, emax, template)
        fmt = 'csky__trials_{:07d}_' \
                'gp_{}_seed_{:04d}_emin{}_emax_{}'
        label = fmt.format (n_trials, template,  s, emin, emax)
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
@click.option ('--n-trials', default=20000, type=int)
@click.option ('--n-jobs', default=10, type=int)
@click.option ('-n', '--n-sig', 'n_sigs', multiple=True, default=[0], type=float)
@click.option ('--gamma', default=2, type=float)
@click.option ('-c', '--cutoff', default=np.inf, type=float, help='exponential cutoff energy (TeV)')      
@click.option ('--poisson/--nopoisson', default=True)
@click.option ('--dry/--nodry', default=False)
@click.option ('--catalog', type=str, default=None)
@click.option ('--seed', default=0)
@click.option('--memory', default=0.5, type=float, help='Requested memory (GB)')
@pass_state
def submit_do_stacking_trials (
        state, n_trials, n_jobs, n_sigs, gamma, cutoff,  poisson,  dry, 
        catalog,  seed, memory):
    ana_name = state.ana_name
    T = time.time ()
    poisson_str = 'poisson' if poisson else 'nopoisson'
    job_basedir = state.job_basedir 
    job_dir = '{}/{}/stacking_trials/T_E{}_{:17.6f}'.format (
        job_basedir, ana_name, int(gamma * 100),  T)
    sub = Submitter (job_dir=job_dir, memory=memory,
        max_jobs=1000, config = submit_cfg_file)
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
                fmt = ' {} do-stacking-trials --catalog={} --n-trials={}' \
                        ' --n-sig={} --gamma={:.3f} --cutoff={}' \
                        ' --{} --seed={}  '
  
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
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels)
    else:
        sub.submit_npx4 (commands, labels)


@cli.command ()
@click.option ('--n-jobs', default=10, type=int)
@click.option ('--cpus', default=1, type=int)
@click.option ('-n', '--n-sig', default=0, type=float)
@click.option ('--gamma', default=2, type=float)
@click.option ('--dec_deg', 'dec_deg',  type=float, default= 0 )
@click.option ('--dry/--nodry', default=False)
@click.option ('--seed', default=0)
@click.option('--nside', default=128, type=int)
@click.option('--poisson/--nopoisson', default=True,
              help='toggle possion weighted signal injection')
@click.option('--fit/--nofit', default=False,
              help='Use Chi2 Fit or not for the bg trials at each declination')
@click.option('--memory', default=3, type=float, help='Requested memory (GB)')
@pass_state
def submit_do_sky_scan_trials (
        state,  n_jobs, cpus, n_sig, gamma, nside, poisson, fit,
         dec_deg, dry, 
        seed, memory):
    ana_name = state.ana_name
    T = time.time ()

    poisson_str = 'poisson' if poisson else 'nopoisson'
    fit_str = 'fit' if fit else 'nofit'

    job_basedir = state.job_basedir 
    job_dir = '{}/{}/skyscan_trials/T_E{}_n_sig_{}_{:17.6f}'.format (
        job_basedir, ana_name, int(gamma * 100), n_sig,  T)
    sub = Submitter (job_dir=job_dir, memory=memory, ncpu=cpus,
        max_jobs=1000, config = submit_cfg_file)
    commands, labels = [], []
    trial_script = os.path.abspath('trials.py')
    for i in range (n_jobs):
        s = i + seed
        fmt = (
            ' {} do-sky-scan-trials --dec_deg={:+08.3f}'
            ' --n-sig={} --gamma={:.3f} --seed={} --cpus={}'
            ' --nside={} --{} --{}'
        )
        command = fmt.format(
            trial_script,  dec_deg,
            n_sig, gamma, s, cpus,
            nside, poisson_str, fit_str,
        )

        fmt = 'csky__scan_dec_{:+08.3f}___n_sig_{:08.3f}__' \
                'gamma_{:.3f}_seed_{:04d}'
        label = fmt.format (dec_deg, n_sig, gamma,
                             s)
        commands.append (command)
        labels.append (label)
    sub.dry = dry
    reqs = '(Machine != "cobol85.private.pa.umd.edu")'
    print(hostname)
    if 'condor00' in hostname:
        sub.submit_condor00 (commands, labels, reqs=reqs)
    else:
        sub.submit_npx4 (commands, labels)

if __name__ == '__main__':
    exe_t0 = now ()
    print ('start at {} .'.format (exe_t0))
    cli ()
