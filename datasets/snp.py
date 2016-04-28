'''
SNP dataset class.
'''

from collections import OrderedDict
import numpy as np
from scipy.io import loadmat
from . import BasicDataset, Dataset
from utils.tools import (
    warn_kwargs
)


class SNP(BasicDataset):
    '''SNP dataset class.

    Currently only handled continuous preprocessed data.
    Discrete data TODO
    '''
    def __init__(self, source=None, name='snp', mode='train', convert_one_hot=True, idx=None, **kwargs):
        '''Initialize the SNP dataset.

        Arguments:
            source: str. Path to source file.
            name: str. ID of dataset.
            idx: (Optional) list. List of indices for train/test/validation
                split.
        '''
        #kwargs = super(SNP, self).__init__(name=name, **kwargs)
        if source is None:
            raise ValueError('No source provided')

        # Fetch SNP data from "source"
        X, Y = self.get_data(source)
        
        data = {name: X, 'label': Y}
        import ipdb
        ipdb.set_trace()
        balance = False
        if idx is not None:
            balance=True
            data[name] = data[name][idx]
            data['label'] = data['label'][idx]
        
        distributions = {name: 'gaussian', 'labels': 'multinomial'}
        super(SNP, self).__init__(data, name=name, balance=balance, distributions=distributions,  mode=mode, **kwargs)

        self.mean_image = self.data[name].mean(axis=0)

    def get_data(self, source):
        '''Fetch the data from source.
        Arguments:
           source: dict. file names of genetic data and labels
                  {'snp' key for genetic data
                    'labels' key for diagnosis }
                    
        Genetic data is in the matrix format with size Subjec*SNP
        SNP can be either preprocessed or notprocessed
        Labels is a vector with diagnosis info
        Patients are coded with 1 and health control coded with 2
        '''
        from utils.tools import get_paths
        data_path = get_paths()['$snp_data']
        print('Loading genetic data from %s' % data_path)
        X = loadmat(data_path + '/' + source['snp'])
        Y = loadmat(data_path + '/' + source['labels'])
        X = np.float32(X[X.keys()[2]])
        Y = np.float32(Y[Y.keys()[0]])
        Y.resize(max(Y.shape,))
        return X, Y

    """def randomize(self):
        '''Randomize the dataset.'''
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx,:]
        self.O = self.O[rnd_idx]"""

    def reset(self):
        '''Reset the iterator'''
        self.pos = 0
        if self.shuffle:
            self.randomize()

    """def next(self, batch_size=None):
        '''Iterate the data.'''

        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()
            raise StopIteration

        x = self.X[self.pos:self.pos + batch_size]
        y = self.O[self.pos:self.pos + batch_size]

        rval = OrderedDict(pos=self.pos)

        if self.pos + batch_size > self.n:
            self.pos = -1
        else:
            self.pos += batch_size

        rval[self.name] = x
        rval['labels'] = y

        return rval"""
