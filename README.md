# DNNCascade

Analysis Scripts for DNNCascade Source Searches:
Analysis wiki: https://wiki.icecube.wisc.edu/index.php/Cascade_Neutrino_Source_Dataset/Analyses

Requirements: 

* cvmfs with python 3 `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
* csky tag v1.1.12
* click (any version should work, but ran with version 7.1.2)
* pandas (any version should work, but ran with version 1.1.1 )
* numpy (any version should work, but ran with version 1.18.5 [dep. of csky])
* scipy (any version should work, but ran with version 1.7.3 [dep. of csky])
* matplotlib (ran with version 3.5.1, submitter might need version < 3.1 + MPLBACKEND='AGG'  [dep. of csky])
* Submitter (https://github.com/ssclafani949/Submitter) 

A version of this virtual environment is saved at /data/ana/analyses/NuSources/2021_DNNCasacde_analyses/venv

Generated trials have been saved to `/data/ana/analyses/NuSources/2021_DNNCascade_analyses/baseline_analysis`.
Note that these were generated with csky version v1.1.7 prior to finding the GRL and the two csky bugs that were fixed in PR #87 and release v1.1.12. Checks were performed which indicate that no re-run of background trials is necessary.

File Structure:
Config.py
  This script sets the `job_base` which can be left as baseline_analysis.  It also sets different directories to save everything.  These can be adjusted but will default by creating `/data/user/USERNAME/data/analyses/JOB_BASE/` directory where all files will be read and saved unless otherwise specified.
  
trials.py
  This script has functions to run trials or compute sensitivity, for PS, Stacking and Templates, these can be done at the individiual trial level, using the `do-X-trials` functions, or computing a senstivity from scratch using `do-X-sens` .  The sens functions are useful for quick checks that do not require a lot of background trials.
  
submit.py
  This script maps to the functions in `trials.py` and controls the submission script writing for each function.  If you are using NPX or UMD cluster you can call these functions from `submit-1` on NPX and it will create the relavant dagman and submit this. 
  
submitter_config
  This is a small config file that is run on each job that is submitted.  It currently loads cvmfs and then loads a virtual environment with relavant software
 
 unblind.py
  The script that will be used to unblind and run correlated trials.

Setup: 
Trials are run either on cobalt, npx, or on local machines with cvmfs and virtual environment.  IceRec or combo is not required.  To setup call cvmfs via `eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh)` while this is loaded create a virtual environment to insall packages using `python -m venv /path/to/virtualenvironemnt`
In the virtual environment install the required software.

Submitter should not be necessary for small scale testing but instructions are below:

Submitter requires a config file that will load cvmfs and the virtual environemnt for each job.  This is included as submitter_config but can be set elsewhere.  This config should contain the two lines:

```
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh`
source  ~/path/to/venv/bin/activate
```
To submit a job an example would be:

`python submit.py submit-do-ps-trials --dec_degs 30 --n-trials 1000 --n-jobs 1 --gamma 2 --n-sig 0 --n-sig 10`

This will submit 1 job to run 1000 bg trials and 1000 trials with n-sig of 0 and a nsig of 10.  The default paramaters will be to run enough trials for analysis calculations

The submit.py will call the above function for many different arguments assisting with job submital.  If called from submit-1
the submitter package will create the dag and start it.  To create the dag, but not run it right away, pass the --dry keyword.


Analysis Examples:
To keep all the analyses separate and consistant all folders are created based on a job_base, and username
these can be edited in config.py.  The default 'baseline' will run all trials with baseline MC.  A jobbase with systematics in the name will run with full range of systematics MC.  The current setup will run either on NPX/Cobalts or on UMD cluster 'condor00'.


## Run PS Trials on Cobalt

All jobs should run easily on cobal machines without using too many resources.  Submittal is mostly for large scale trial production.

eg, run 100 trials at dec = -30  injecting 25 events

`python trials.py do-ps-trials --cpus N --dec_deg -30 --n-trials 100 --n-sig 25`


eg, run calculate sensitvity at the same declination but run all signal and background trials

`python trials.py do-ps-sens --cpus N --dec_deg -30 --n-trials 1000 `

We can set the gamma and cutoff via the `--gamma` and `--cutoff` flags (defaults are gamma=2 and cutoff=inf).
To calculate a discovery potential `--nsigma N` can be passed, this will
automatically set the threshold to be 50% of background trials.
The same steps can be performed with the corresponding `do_XX_YY` functions for stacking, templates, skysca

## Combine PS trials
Once all the background trials are created we need to combine them into one nested dictionary for all parameters:

`python trials.py collect-ps-bg --nofit --nodist`

`python trials.py collect-ps-sig`

We add the `--nofit --nodist` flags to the background trial collection to collect the raw trials.

## Calculate sensitvity

From those combined files we will calculate ps senstivity at each declination for the given gamma and cutoff:
`python trials.py find-ps-n-sig --gamma X --cutoff inf`

If we instead want to compute the discovery potential for a given sigma, we can additionally set the `--nsigma` flag.
To compute the 3-sigma discovery potential we can do the following:
`python trials.py find-ps-n-sig --gamma X --cutoff inf --nsigma 3`

Note that this uses csky's method `csky.bk.get_best` under the hood. This method will try to find the closest background and signal trials to the ones defined via the flags. This functionality is very useful when interpolating on a grid of gamma, dec or cutoff values, for instance. However, it can be mis-leading when only spot-checks are performed and only trials for a single gamma, declination or cutoff value are computed. Make sure that the trials exist for which the sensitivity is computed!

A similar analysis chain can be performed with the functions for stacking, templates, skyscan. 
Of note is the syntax for templates, which is slightly different and the template must be at the end:

`python trials.py do-gp-trials --n-trials 1000 pi0 `

Possible template arguments are: `pi0`, `kra5`, `kra50`, `fermibubbles`.
For convenience, examples of the analysis chains for the catalog stacking searches and the galactic plane templates are shown below.
       

## Analysis chain for galactic plane templates

For convenience, the analysis chain for a reduced number of trials and signal injections for the galactic plane templates is outlined below:

        # run background trials
        python trials.py do-gp-trials --n-trials 20000 --cpus 12 <template>
        
        # run signal trials (for fermibubbles an additional flag `--cutoff <cutoff>` is needed)
        python trials.py do-gp-trials --n-trials 100 --cpus 12 --n-sig <n-sig> <template>
        
        # collect trials (this collects both signal and background trials)
        python trials.py collect-gp-trials
        
        # find sensitivity (for discovery potential pass flag `--nsigma <N>`)
        python trials.py find-gp-n-sig --nofit 
        
        
Insert each of `[pi0, kra5, kra50, fermibubbles]` for `<template>` and for the fermibubbles each of `[50, 100, 500, inf]` for `<cutoff>`. A reduced set of different `<n-sig>` values for testing could be: `[50, 100, 200, 300]`.


## Analysis chain for stacking analyses

For convenience, the analysis chain for a reduced number of trials and signal injections for the stacking catalogs is outlined below:

        # run background trials
        python trials.py do-stacking-trials --n-trials 1000 --cpus 12 --catalog <catalog>
        
        # run signal trials
        python trials.py do-stacking-trials --n-trials 100 --cpus 12 --gamma 2.0 --n-sig <n-sig> --catalog <catalog>
        
        # collect background trials
        python trials.py collect-stacking-bg
        
        # collect signal trials
        python trials.py collect-stacking-sig
        
        # find sensitivity (for discovery potential pass flag `--nsigma <N>`)
        python trials.py find-stacking-n-sig --nofit 


Insert each of `[snr, pwn, unid]` for `<catalog>`.
A reduced set of different `<n-sig>` values for testing could be: `[10, 20, 50]`.


## Correlated trials for source list

The most significant source from the source list will be reported. In order to perform the trial correction, correlated trials are considered by utilizing csky's `MultiTrialRunner`. 

        # run background trials at exact declinations of sources
        # <source-num> runs from 0 to 108 for each of the 109 sources in the list
        python trials.py do-bkg-trials-sourcelist --n-trials <ntrials> --cpus <ncpus> --sourcenum <source-num>
        
        # collect background trials
        python trials.py collect-bkg-trials-sourcelist
        
        # perform correlated trials
        python trials.py do-correlated-trials-sourcelist --n-trials <ntrials> --cpus <ncpus> 
        
        # collect correlated background trials
        python trials.py collect-correlated-trials-sourcelist


## Correlated trials for Fermibubble cutoffs

In order to perform the trial correction for the most significant cutoff energy for the Fermi bubble template, correlated trials are considered with csky's `MultiTrialRunner`. We can use the trials for each cutoff `[50, 100, 500, inf] TeV` that we've computed before and utilize these for the `MultiTrialRunner` to compute correlated trials.

        # perform correlated trials
        python trials.py do-correlated-trials-fermibubbles --cpus <ncpus> --n-trials <ntrials>
        
        # collect correlated background trials
        python trials.py collect-correlated-trials-fermibubbles
        
## Correlated trials for sky-scan

In order to perform the trial correction for the most significant pixel in the sky, correlated trials are considered with csky's sky scan trial runner. The sky scan trial runner will use the ps trials for each declination that must have been computed and collected prior to runnight the sky-scan trial runner. 

        # perform correlated trials
        python trials.py do-sky-scan-trials --cpus <ncpus>
        
        # collect correlated background trials
        python trials.py collect-sky-scan-trials-bg

Note: we can recalculate the p-value for previously performed scans. This can be useful, if we increase the stats for the
uncorrelated background trials at each declination, for instance, or if we want to change the `--fit` behaviour. 

        # recalculate previous scans with `--nofit`, but now use `--fit`
        python trials.py recalculate-sky-scan-trials --noinputfit --fit
        
        # recalculate previous scans with updated bg trials
        # (We need to add `--overwrite` to overwrite existing files)
        python trials.py recalculate-sky-scan-trials --overwrite


## List of submitter commands for background trials

For convenience, here is a list of submitter commands to run the background trials. 
These are pasted here to illustrate the syntax. Final background trials used in this analysis may
use different seeds and the amount may also vary. Note that the jobs will utilize seeds
from `--seed` to `--seed + --n-jobs -1`. So if submitting 1000 jobs at an initial seed of 0,
the next submission should start at `--seed 1000`.

        # bg trials for ps from -81° to 81° in increments of 2°
        # (1M trials at each dec, ~0.3s/trial)
        python submit.py submit-do-ps-trials --n-trials 20000 --n-jobs 50 --n-sig 0 --seed 0
        
        # bg trials for source list at source declinations
        # (1M trials at each source, ~0.3s/trial)
        python submit.py submit-do-bkg-trials-sourcelist --n-trials 20000 --n-jobs 50 --seed 0
        
        # bg trials for pi0 template (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --memory 2 --seed 0 pi0
        
        # bg trials for kra5 template (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --seed 0 kra5
        
        # bg trials for kra50 template (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --seed 0 kra50
        
        # bg trials for fermibubbles with 50 TeV cutoff (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --memory 2 --cutoff 50 --seed 0 fermibubbles
        
        # bg trials for fermibubbles with 100 TeV cutoff (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --memory 2 --cutoff 100 --seed 0 fermibubbles
        
        # bg trials for fermibubbles with 500 TeV cutoff (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --memory 2 --cutoff 500 --seed 0 fermibubbles
        
        # bg trials for fermibubbles with no cutoff (50M trials, ~0.08s/trial)
        python submit.py submit-do-gp-trials --n-trials 50000 --n-jobs 1000 --memory 2 --cutoff inf --seed 0 fermibubbles
        
        # bg trials for stacking catalog unid (2M trials, ~1s/trial)
        python submit.py submit-do-stacking-trials --n-trials 20000 --n-jobs 100 --catalog unid --seed 0
        
        # bg trials for stacking catalog pwn (2M trials, ~1s/trial)
        python submit.py submit-do-stacking-trials --n-trials 20000 --n-jobs 100 --catalog pwn --seed 0
        
        # bg trials for stacking catalog snr (2M trials, ~1s/trial)
        python submit.py submit-do-stacking-trials --n-trials 20000 --n-jobs 100 --catalog snr --seed 0
        
        
        
Once the uncorrelated trials are done, we can run correlated ones:

        # correlated trials for source list (100K trials, ~30s/trial)
        python submit.py submit-do-correlated-trials-sourcelist --n-trials 1000 --n-jobs 100 --seed 0
        
        # correlated trials for fermibubbles (50M trials, ~0.2s/trial)
        python submit.py submit-do-correlated-trials-fermibubbles --n-trials 50000 --n-jobs 1000 --seed 0
        
        # sky-scan trials (1000 trials, ~12h/trial)
        python submit.py submit-do-sky-scan-trials --n-jobs 1000 --seed 0
        
        
## Unblinding

For the p-value calculation, it's helpful to have an overview of the number of background trials. 
P-values are obtained from the trials directly, rather than using a Chi2 fit.
The number of background trials that go into each analysis may be checked via
        
        python check_trials.py

The script also computes the most significant result that can still be trusted given the number of background trials.
This boundary is defined as the p-value above which `N` background trials remain. Per default, `N` is set to 100,
but it can be adjusted by adding the flag `--trials-after <N>`.
       
In the following, the commands are detailled that may be used to unblind the indiviual searches.
        
        # unblind the source list
        python unblind.py unblind-sourcelist
        
        # unblind the Galactic plane templates
        python unblind.py unblind-gp pi0
        python unblind.py unblind-gp kra5
        python unblind.py unblind-gp kra50
        
        # unblind Fermi bubble template
        python unblind.py unblind-fermibubbles
        
        # unblind stacking catalogs
        python unblind.py unblind-stacking
        
        # unblind skyscan
        python unblind.py unblind-skyscan --cpus 15

The above commands will run the "unblinding" for a given data scramble. 
This can be used to test if everything is working as intended. 
To perform the actual unblinding, the flag `--TRUTH` must be added.





