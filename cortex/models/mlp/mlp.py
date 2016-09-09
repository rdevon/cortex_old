'''Module for MLP model.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
import warnings

from .. import batch_normalization, Cell, dropout, norm_weight
from ...utils import concatenate, floatX


_nonlinearity_dict = dict(
    identity=(lambda x: x),
    sigmoid=T.nnet.sigmoid,
    hard_sigmoid=T.nnet.hard_sigmoid,
    fast_sigmoid=T.nnet.ultra_fast_sigmoid,
    tanh=T.tanh,
    softplus=T.nnet.softplus,
    relu=T.nnet.relu
)

def resolve_nonlinearity(key):
    nonlin = _nonlinearity_dict.get(key)
    if nonlin is None:
        nonlin = eval(key)
    return nonlin

class MLP(Cell):
    '''Multilayer perceptron model.

    Attributes:
        dim_in (int): input dimension.
        dim_out (int): output dimension.
        dim_h (Optional[int]): dimension of hidden layer.
        dim_hs (Optional[list]): for multiple hidden layers.
        h_act (str): hidden activation string.
        out_act (str): output activation string.

    '''
    _required = ['dim_in', 'dim_out']
    _options = {'dropout': False, 'weight_noise': 0,
                'batch_normalization': False}
    _args = ['dim_in', 'dim_out', 'dim_hs', 'h_act', 'out_act']
    _dim_map = {
        'X': 'dim_in',
        'input': 'dim_in',
        'Y': 'dim_out',
        'output': 'dim_out',
        'Z': 'dim_out'
    }
    _weights = ['weights']

    def __init__(self, dim_in, dim_out, dim_h=None, n_layers=None, dim_hs=None,
                 h_act='sigmoid', out_act=None, name='MLP', **kwargs):
        '''Init function for MLP.

        Args:
            dim_in (int): input dimension.
            dim_out (int): output dimension.
            dim_h (Optional[int]): dimension of hidden layer.
            n_layers (int): number of output and hidden layers.
            dim_hs (Optional[list]): for multiple hidden layers.
            h_act (str): hidden activation string.
            out_act (Optional[str]): output activation string. If None, then set
                to `h_act`.
            name (str): default name id of cell.
            **kwargs: optional init keyword arguments.

        '''
        if n_layers is not None and n_layers < 1:
            raise ValueError('`n_layers must be > 0')
        self.dim_in = dim_in
        self.dim_out = dim_out
        if out_act is None: out_act = h_act
        self.out_act = resolve_nonlinearity(out_act)
        self.h_act = resolve_nonlinearity(h_act)

        # Various means to get the hidden layers.
        if dim_hs is None and dim_h is None and n_layers is None:
            dim_hs = []
        elif dim_hs is None and dim_h is None:
            dim_hs = [dim_out] * (n_layers - 1)
        elif dim_hs is None:
            dim_hs = [dim_h] * (n_layers - 1)
        elif dim_h is None and n_layers is None:
            pass
        elif dim_h is None:
            assert len(dim_hs) == n_layers
        elif dim_hs is None and n_layers is None:
            dim_hs = []

        self.dim_hs = dim_hs
        self.n_layers = len(dim_hs) + 1
        super(MLP, self).__init__(name=name, **kwargs)

    def init_params(self, weight_scale=1e-3):
        self.params = OrderedDict()

        dim_ins = [self.dim_in] + self.dim_hs
        dim_outs = self.dim_hs + [self.dim_out]
        weights = []
        biases = []

        for dim_in, dim_out in zip(dim_ins, dim_outs):
            W = norm_weight(dim_in, dim_out, scale=weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)
            weights.append(W)
            biases.append(b)

        self.params['weights'] = weights
        self.params['biases'] = biases

    def get_params(self):
        params = zip(self.weights, self.biases)
        params = [i for sl in params for i in sl]
        return super(MLP, self).get_params(params=params)

    def feed(self, X, *params):
        '''feed forward MLP.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            OrderedDict: results at every layer.

        '''
        session = self.manager._current_session

        params = list(params)
        outs = OrderedDict(X=X)
        outs['input'] = X
        for l in xrange(self.n_layers):
            if self.batch_normalization:
                self.logger.debug('Batch normalization on layer %d' % l)
                X = batch_normalization(X, session=session)

            W = params.pop(0)
            b = params.pop(0)
            preact = T.dot(X, W) + b

            if l < self.n_layers - 1:
                X = self.h_act(preact)
                outs['G_%d' % l] = preact
                outs['H_%d' % l] = X

                if self.dropout and self.noise_switch():
                    epsilon = params.pop(0)
                    self.logger.debug('Adding dropout to layer {layer} for MLP '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = dropout(X, self.h_act, self.dropout, self.trng,
                                epsilon=epsilon)
            else:
                X = self.out_act(preact)
                outs['Z'] = preact
                outs['Y'] = X
                outs['output'] = X

        assert len(params) == 0, params
        return outs
    '''
    def get_n_params(self):
        n_params = self.n_params
        if self.dropout and self.noise_switch(): n_params += self.n_layers - 1
        return n_params
    '''

    def _feed(self, X, *params):
        params = list(params)
        if self.dropout and self.noise_switch():
            if X.ndim == 1:
                size = None
            elif X.ndim == 2:
                size = (X.shape[0],)
            elif X.ndim == 3:
                size = (X.shape[0], X.shape[1])
            elif X.ndim == 4:
                size = (X.shape[0], X.shape[1], X.shape[2])
            else:
                raise TypeError

            epsilons = self.dropout_epsilons(size)
            params = self.get_epsilon_params(epsilons, *params)

        return self.feed(X, *params)

    def get_epsilon_params(self, epsilons, *params):
        if self.dropout and self.noise_switch():
            new_params = []
            for l in xrange(self.n_layers - 1):
                new_params += params[2*l:2*(l+1)]
                new_params.append(epsilons[l])
            new_params += params[2*(l+1):]
            assert len(new_params) == len(params) + self.n_layers - 1

            return new_params
        else:
            return params

    def dropout_epsilons(self, size):
        epsilons = []

        for dim_h in self.dim_hs:
            shape = size + (dim_h,)
            eps = self.trng.binomial(shape, p=1-self.dropout, n=1, dtype=floatX)
            epsilons.append(eps)

        return epsilons


_classes = {'MLP': MLP}