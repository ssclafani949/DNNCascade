# DNNCascade

Analysis Scripts for DNNCascade Source Searches:
Analysis wiki: https://wiki.icecube.wisc.edu/index.php/Cascade_Neutrino_Source_Dataset/Analyses

Requirements: 

* cvmfs with python 3 `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
* csky tag v1.1.6
* click (any version should work, but ran with version 7.1.2)
* numpy (any version should work, but ran with version 1.18.5)
* pandas (any version should work, but ran with version 1.1.1)
* Submitter (https://github.com/ssclafani949/Submitter) 

A version of this virtual environment is saved at /data/ana/analyses/NuSources/2021_DNNCasacde_analyses/venv

Generated trials have been saved to `/data/ana/analyses/NuSources/2021_DNNCascade_analyses/baseline_analysis`

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

`python submit.py submit-do-ps-trials --dec_degs 30 --n_trials 1000 --n-jobs 1 --gamma 2 --n-sig 0 --n-sig 10`

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
        
Insert each of `[pi0, kra5, kra50, fermibubbles]` for `<template>` and for the fermibubbles each of `[50, 100, 500, None]` for `<cutoff>`. Note that `None` can't be passed in. However, this is the default value for `--cutoff`, so the flag `--cutoff` does not need to be set in this case. A reduced set of different `<n-sig>` values for testing could be: `[50, 100, 200, 300]`.


## Analysis chain for stacking analyses

For convenience, the analysis chain for a reduced number of trials and signal injections for the stacking catalogs is outlined below:

        # run background trials
        python trials.py do-stacking-trials --n-trials 1000 --cpus 12 --catalog <catalog>
        
        # run signal trials
        python trials.py do-stacking-trials --n-trials 100 --cpus 12- -gamma 2.0 --n-sig <n-sig> --catalog <catalog>
        
        # collect background trials
        python trials.py collect-stacking-bg
        
        # collect signal trials
        python trials.py collect-stacking-sig
        
        # find sensitivity (for discovery potential pass flag `--nsigma <N>`)
        python trials.py find-stacking-n-sig --nofit 


Insert each of `[snr, pwn, unid]` for `<catalog>`.
A reduced set of different `<n-sig>` values for testing could be: `[10, 20, 50]`.

