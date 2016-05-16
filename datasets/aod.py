'''
Module for AOD analysis functions.
'''

import logging
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

from mvpa2.mappers.detrend import poly_detrend
from mvpa2.datasets import dataset_wizard

from utils import ols


def regress_to_stims(activations, targets, novels):
    dt = targets.shape[0]

    assert targets.shape == novels.shape
    assert activations.shape[1] % dt == 0

    n_subjects = activations.shape[1] // dt
    targets_novels = np.concatenate([targets[:, None], novels[:, None]], axis=1)

    target_ttests = []
    novel_ttests = []
    for i in xrange(activations.shape[0]):
        betas = np.zeros((n_subjects, 2))
        for s in xrange(n_subjects):
            stats = ols.ols(activations[i, dt * s : dt * (s + 1)], 
                            targets_novels)
            betas[s] = stats.b[1:]

        target_ttests.append(ttest_1samp(betas[:, 0], 0))
        novel_ttests.append(ttest_1samp(betas[:, 1], 0))

    return target_ttests, novel_ttests

def detrend(acts, poly=4):
   tlen = 249
   features = acts.shape[-1]
   print('Detrending...')
   if len(acts.shape) == 2:    
      subjects = acts.shape[0] / tlen
      x = np.arange(subjects)
      chunks = np.repeat(x, tlen)
      ds = dataset_wizard(acts, chunks=chunks)
      poly_detrend(ds, chunks_attr='chunks', polyord=poly)      
      return ds.samples
   else:
      subjects = acts.shape[0]
      for s in range(subjects):
         ds = dataset_wizard(acts[s,:,:])
         poly_detrend(ds, polyord=poly)
         acts[s,:,:] = ds.samples
      return acts
