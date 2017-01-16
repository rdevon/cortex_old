'''Module for MLP model.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet.bn import batch_normalization
import warnings

from .. import Cell, dropout, norm_weight
from ...utils import concatenate, floatX


_nonlinearity_dict = dict(
    identity=(lambda x: x),
    sigmoid=T.nnet.sigmoid,
    hard_sigmoid=T.nnet.hard_sigmoid,
    fast_sigmoid=T.nnet.ultra_fast_sigmoid,
    tanh=T.tanh,
    softplus=T.nnet.softplus,
    relu=T.nnet.relu,
    leaky_relu=lambda x: T.nnet.relu(x, alpha=0.01)
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
                'batch_normalization': False, 'bn_mean_only': False,
                'weight_normalization': False}
    _args = ['dim_in', 'dim_out', 'dim_hs', 'h_act', 'out_act', 'out_scale', 'dropout']
    _dim_map = {
        'X': 'dim_in',
        'input': 'dim_in',
        'Y': 'dim_out',
        'output': 'dim_out',
        'Z': 'dim_out'
    }
    _weights = ['weights']

    def __init__(self, dim_in, dim_out, dim_h=None, n_layers=None, dim_hs=None,
                 h_act='sigmoid', out_act=None, out_scale=None, name='MLP',
                 **kwargs):
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
        self.out_scale = out_scale

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
        super(MLP, self).init_params()

        dim_ins = [self.dim_in] + self.dim_hs
        dim_outs = self.dim_hs + [self.dim_out]
        weights = []
        biases = []
        if self.batch_normalization:
            gammas = []
            betas = []
            
        if self.weight_normalization:
            gs = []

        for dim_in, dim_out in zip(dim_ins, dim_outs):
            W = norm_weight(dim_in, dim_out, scale=weight_scale, ortho=False)
            b = np.zeros((dim_out,))
            weights.append(W)
            biases.append(b)
            if self.batch_normalization:
                gamma = np.ones((dim_in,))
                beta = np.zeros_like(gamma)
                gammas.append(gamma)
                betas.append(beta)
            if self.weight_normalization:
                g = np.float32(1.)
                gs.append(g)

        self.params['weights'] = weights
        self.params['biases'] = biases
        if self.batch_normalization:
           self.params['gammas'] = gammas
           self.params['betas'] = betas
        if self.weight_normalization:
            self.params['gs'] = gs

    def get_params(self):
        param_list = [self.weights, self.biases]
        if self.batch_normalization:
            param_list = param_list + [self.gammas, self.betas]
        if self.weight_normalization:
            param_list.append(self.gs)
        params = zip(*param_list)
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
            W = params.pop(0)
            b = params.pop(0)

            if self.batch_normalization:
                gamma = params.pop(0)
                beta = params.pop(0)
                self.logger.debug('Batch normalization on layer %d' % l)
                if X.ndim == 2:
                    mean = X.mean(0, keepdims=True)
                    std = X.std(0, keepdims=True)
                elif X.ndim == 3:
                    mean = X.mean((0, 1), keepdims=True)
                    std = X.std((0, 1), keepdims=True)
                else:
                    raise ValueError()
                if self.bn_mean_only:
                    std = T.ones_like(std)
                    gamma = T.ones_like(gamma) + 0. * gamma
                else:                    
                    std = T.sqrt(std ** 2 + 1e-6)
                X = batch_normalization(inputs=X, gamma=gamma, beta=beta,
                                        mean=mean, std=std, mode='high_mem')
            if self.weight_normalization:
                g = params.pop(0)
                W = g * W / (W ** 2).sum(axis=1, keepdims=True)

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
                if self.out_scale is not None: X = X * self.out_scale
                outs['Z'] = preact
                outs['Y'] = X
                outs['output'] = X

        assert len(params) == 0, params
        return outs

    def _feed(self, X, *params):
        params = list(params)
        if self.dropout and self.noise_switch():
            if X.ndim == 1:
                size = tuple()
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
            ppl = 2
            if self.batch_normalization:
                ppl += 2
            if self.weight_normalization:
                ppl += 1
            
            new_params = []
            for l in xrange(self.n_layers - 1):
                new_params += params[ppl*l:ppl*(l+1)]
                new_params.append(epsilons[l])
            new_params += params[ppl*(self.n_layers-1):]
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