''' Tests for MRI and fMRI data.

'''

from os import path
import warnings

import cortex
from cortex.utils.tools import resolve_path
from cortex.utils import floatX, logger as cortex_logger


cortex_logger.set_stream_logger(2)
manager = cortex._manager
VBM_path = resolve_path('$data/neuroimaging/VBM_test/VBM.yaml')
AOD_path = resolve_path('$data/neuroimaging/AOD_test/AOD.yaml')

if not path.isfile(VBM_path):
    warnings.warn('Cannot test MRI dataset. `%s` not found' % VBM_path)
    exit()

if not path.isfile(AOD_path):
    warnings.warn('Cannot test fMRI dataset. `%s` not found' % AOD_path)
    exit()

def _test_class(c='MRI'):
    if c == 'MRI':
        p = VBM_path
    elif c in ['FMRI_IID', 'FMRI']:
        p = AOD_path
    else:
        raise TypeError
    cortex.prepare_data(c, source=p)

def _test_split(c='MRI'):
    if c == 'MRI':
        p = VBM_path
    elif c in ['FMRI_IID', 'FMRI']:
        p = AOD_path
    else:
        raise TypeError
    cortex.prepare_data_split(c, source=p, split=[0.7, 0.2, 0.1])

def test_classes():
    for c in ['MRI', 'FMRI_IID', 'FMRI']:
        _test_class(c)
        _test_split(c)