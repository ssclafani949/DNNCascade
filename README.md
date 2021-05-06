# DNNCascade
Analysis Scripts for DNNCascade Source Search

Requirements csky, click, submitter

Usage examples.

#Run PS Trials on Cobalt
eg, run 100 trials at dec = -30 with the NN based PDFs injecting 25 events

`python trials.py do-ps-trials --cpus N --nn --dec_deg -30 --n-trials 100 --n-sig 25`

The submit.py will call the above function for many different arguments assisting with job submital.  If called from submit-1
the submitter package will create the dag and start it.

