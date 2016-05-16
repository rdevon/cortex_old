'''Convenience class for preprocessing data.

Not meant to be general, but feel free to subclass if it's useful
'''

from collections import OrderedDict
import theano

from cortex.utils import floatX


class Preprocessor(object):
    '''Preprocessor class.

    Attributes:
        processes: OrderedDict, preprocessing steps in order.
    '''

    keys = ['center']
    keyvals = []

    def __init__(self, proc_list):
        '''Init method.

        Args:
            proc_list: list.
        '''
        self.processes = OrderedDict()
        for proc in proc_list:
            if not isinstance(proc, list):
                if proc not in self.keys:
                    raise ValueError('Processing step, %s, not supported' % proc)
                self.processes[proc] = True
            else:
                assert len(proc) == 2
                if proc[0] not in self.keyvals:
                    raise ValueError('Processing step, %s, not supported' % proc[0])
                self.processes[proc[0]] = proc[1]

    def center(self, X, data_iter=None):
        '''Center input.'''
        assert data_iter is not None
        print 'Centering input with {mode} dataset mean image'.format(mode=data_iter.mode)
        X_mean = theano.shared(data_iter.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
        return X_i

    def __call__(self, X, data_iter=None):
        '''Run preprocessing.'''
        for process in self.processes:
            if process == 'center':
                X = self.center(X, data_iter=data_iter)
        return X
