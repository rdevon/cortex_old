'''Module for MLP model.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
import warnings

from . import distributions
from . import Cell, norm_weight, dropout
from ..manager import get_manager
from ..utils import concatenate, floatX


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
    _options = {'dropout': False, 'weight_noise': 0}
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

    def _feed(self, X, *params):
        '''feed forward MLP.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            OrderedDict: results at every layer.

        '''
        params = list(params)
        outs = OrderedDict(X=X)
        outs['input'] = X
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise and self.noise_switch():
                self.logger.debug(
                    'Using weight noise in layer %d for MLP %s' % (l, self.name))
                W_n = W + self.trng.normal(
                    avg=0., std=self.weight_noise, size=W.shape)
                preact = T.dot(X, W_n) + b
            else:
                preact = T.dot(X, W) + b

            if l < self.n_layers - 1:
                X = self.h_act(preact)
                outs['G_%d' % l] = preact
                outs['H_%d' % l] = X
                if self.dropout and self.noise_switch():
                    self.logger.debug('Adding dropout to layer {layer} for MLP '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = dropout(X, self.h_act, self.dropout, self.trng)
            else:
                X = self.out_act(preact)
                outs['Z'] = preact
                outs['Y'] = X
                outs['output'] = X

        assert len(params) == 0, params
        return outs


class DistributionMLP(Cell):
    '''MLP with a distribution as the output.

    Attributes:
        distribution (Distribution): distribution of output. Used for
            sampling, density calculations, etc.
        mlp (MLP): multi-layer perceptron. Used for feed-forward operation.

    '''

    _required = ['distribution_type']
    _components = {
        'mlp': {
            'cell_type': 'MLP',
            '_required': {'out_act': 'identity'},
            '_passed': ['dim_in', 'dim_h', 'n_layers', 'dim_hs', 'h_act']
        },
        'distribution': {
            'cell_type': '&distribution_type',
            '_required': {'conditional': True},
            '_passed': [
                'dim', 'has_kl', 'neg_log_prob', 'kl_divergence', 'simple_sample']
        },
    }
    _links = [('mlp.output', 'distribution.input')]
    _dim_map = {
        'input': 'dim_in',
        'output': 'dim',
        'P': 'dim',
        'samples': 'dim',
        'X': 'dim_in',
        'Y': 'dim',
        'Z': 'dim',
    }
    _dist_map = {'P': 'distribution_type'}
    _costs = {
        'nll': '_cost',
        'negative_log_likelihood': '_cost'
    }
    _sample_tensors = ['P']

    def __init__(self, distribution_type, name=None, **kwargs):
        if name is None:
            name = '%s_%s' % (distribution_type, 'MLP')
        self.distribution_type = distribution_type
        super(DistributionMLP, self).__init__(name=name, **kwargs)

    @classmethod
    def set_link_value(C, key, distribution_type=None, **kwargs):
        manager = get_manager()
        if key != 'output':
            return super(DistributionMLP, C).set_link_value(
                key, distribution_type=distribution_type, **kwargs)

        if distribution_type is None:
            raise ValueError
        DC = manager.resolve_class(distribution_type)
        return DC.set_link_value(key, dim=dim, **kwargs)

    @classmethod
    def get_link_value(C, link, key):
        manager = get_manager()
        if key != 'output':
            return super(DistributionMLP, C).get_link_value(link, key)
        if link.distribution is None:
            raise ValueError
        DC = manager.resolve_class(link.distribution)
        return DC.get_link_value(link, key)

    def _feed(self, X, *params):
        outs = self.mlp._feed(X, *params)
        Y = outs['Y']
        outs['output'] = self.distribution(Y)
        outs['P'] = outs['output']
        return outs

    def _cost(self, X=None, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[self.name + '.' + 'P']
        return self.distribution._cost(X=X, P=P)

    def generate_random_variables(self, shape, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[self.name + '.' + 'P']

        return self.distribution.generate_random_variables(shape, P=P)

    def _sample(self, epsilon, P=None):
        session = self.manager.get_session()
        if P is None:
            if _p(self.name, 'P') not in session.tensors.keys():
                raise TypeError('%s.P not found in graph nor provided'
                                % self.name)
            P = session.tensors[_p(self.name, 'P')]
        return self.distribution._sample(epsilon, P=P)

_classes = {'MLP': MLP, 'DistributionMLP': DistributionMLP}