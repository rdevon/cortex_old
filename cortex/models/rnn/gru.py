"""
Module for GRU layers
"""

import copy
from collections import OrderedDict
import numpy as np
import random
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from .. import ortho_weight
from .. import Cell
from ...utils import concatenate, floatX, slice2, tslice


class GRU(Cell):
    _required = ['dim_h']
    _options = {'weight_noise': False}
    _args = ['dim_h']
    _weights = ['Ura', 'Urb']
    _dim_map = {
        'input': 'dim_h',
        'output': 'dim_h'
    }

    def __init__(self, dim_h, name='RU', **kwargs):
        self.dim_h = dim_h
        super(GRU, self).__init__(name=name, **kwargs)

    def init_params(self):
        '''Initialize RNN parameters.

        '''
        Ura = np.concatenate([ortho_weight(self.dim_h),
                              ortho_weight(self.dim_h)], axis=1)
        Urb = ortho_weight(self.dim_h)
        self.params = OrderedDict(Ura=Ura, Urb=Urb)
    
    def get_gates(self, x):
        '''Split gates.
        Args:
            x (T.tensor): input
        Returns:
            T.tensor: reset gate.
            T.tensor: update gate.
        '''
        r = T.nnet.sigmoid(tslice(x, 0, x.shape[x.ndim-1] // 2))
        u = T.nnet.sigmoid(tslice(x, 1, x.shape[x.ndim-1] // 2))
        return r, u
    
    def _recurrence(self, m, y_a, y_i, h_, Ura, Urb):
        '''Recurrence function.

        Args:
            m (T.tensor): masks.
            y_a (T.tensor): aux inputs.
            y_i (T.tensor): inputs.
            h_ (T.tensor): recurrent state.
            W (theano.shared): recurrent weights.

        Returns:
            T.tensor: next recurrent state.

        '''
        preact = T.dot(h_, Ura) + y_a
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + y_i
        h = T.tanh(preactx)
        h = u * h + (1. - u) * h_
        h = m * h + (1 - m) * h_
        return h

    def _feed(self, X, M, H0, *params):
        n_steps = X.shape[0]
        
        Xa = slice2(X, 0, 2 * X.shape[X.ndim-1] // 3)
        Xb = slice2(X, 2 * X.shape[X.ndim-1] // 3, X.shape[X.ndim-1])
        seqs         = [M[:, :, None], Xa, Xb]
        outputs_info = [H0]
        non_seqs     = params

        h, updates = theano.scan(
            self._recurrence,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=self.name + '_recurrent_steps',
            n_steps=n_steps)

        return OrderedDict(Xa=Xa, Xb=Xb, H=h, updates=updates)
    
_classes = {'GRU': GRU}
