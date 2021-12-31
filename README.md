# DNNCascade

Analysis Scripts for DNNCascade Source Searches:
Analysis wiki: https://wiki.icecube.wisc.edu/index.php/Cascade_Neutrino_Source_Dataset/Analyses

Requirements: 

* cvmfs with python 3 `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
* csky tag v1.1.6
* click (any version should work, but ran with version 7.1.2)
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

To calculate a discovery potential `--nsigma N` can be passed, this will
automatically set the threshold to be 50% of background trials.
The same steps can be performed with the corresponding `do_XX_YY` functions for stacking, templates, skysca

## Combine PS trials
Once all the background trials are created we need to combine them into one nested dictionary for all paramaters:

`python trials.py collect-ps-bg --fit --dist`
`python trials.py collect-ps-sig`

## Calculate sensitvity

From those combined files we will calculate ps senstivity at each declination for the given gamma, cutoff, and nsigma (leaving nsigma as None will calculate sensitvity)


`python trials.py find-ps-n-sig --gamma X --cutoff np.inf --nsigma None`

A similar analysis chain can be performed with the functions for stacking, templates, skyscan

Of note is the syntax for templates has the template is slightly different and the template must be at the end:

`python trials.py do-gp-trials --n-trials 1000 pi0 `
