# DNNCascade

Analysis Scripts for DNNCascade Source Search

Requirements: 

cvmfs with python 3 `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
csky version XXXXXX
click (any version should work, but ran with version 7.1.2)
Submitter (https://github.com/ssclafani949/Submitter) 


Setup: 
Trials are run either on cobalt, npx, or on local machines with cvmfs and virtual environment.  IceRec or combo is not required.  To setup call cvmfs via `source /cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh` while this is loaded create a virtual environment to insall packages using `python -m venv /path/to/virtualenvironemnt`
In the virtual environment install the required software.

Submitter should not be necessary for small scale testing but instructions are below:
Submitter requires a config file that will load cvmfs and the virtual environemnt for each job.  The default location of this file is ~/.bashrc_condor but can be set elsewhere.  This config should contain the two lines:

```
eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/setup.sh`
source  ~/path/to/venv/bin/activate
```


Usage examples.

## Run PS Trials on Cobalt
eg, run 100 trials at dec = -30 with the NN based PDFs injecting 25 events

`python trials.py do-ps-trials --cpus N --nn --dec_deg -30 --n-trials 100 --n-sig 25`

The submit.py will call the above function for many different arguments assisting with job submital.  If called from submit-1
the submitter package will create the dag and start it.

