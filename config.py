# config.py
import socket
import numpy as np
import csky as cy

hostname = socket.gethostname()
print('Hostname: {}'.format(hostname))
job_base = 'baseline_all'
#job_base = 'systematics_full'
if 'condor00' in hostname or 'cobol' in hostname or 'gpu' in hostname:
    print('Using UMD')
    repo = cy.selections.Repository(local_root='/data/i3store/users/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/i3store/users/ssclafani/data/analyses/{}'.format(job_base))
    job_basedir = '/data/i3home/ssclafani/submitter_logs'
else:
    repo = cy.selections.Repository(local_root='/data/user/ssclafani/data/analyses')
    ana_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses')
    base_dir = cy.utils.ensure_dir('/data/user/ssclafani/data/analyses/{}'.format(job_base))
    ana_dir = '{}/ana'.format (base_dir)
    job_basedir = '/scratch/ssclafani/' 

