'''Module for MLP model.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
import warnings

from . import distributions
from . import Cell, norm_weight, resolve_class
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
    _options = {'dropout': False, 'weight_noise': False}
    _args = ['dim_in', 'dim_out', 'n_layers', 'dim_hs', 'h_act', 'out_act']
    _arg_map = {
        'X': 'dim_in',
        'input': 'dim_in',
        'Y': 'dim_out',
        'output': 'dim_out',
        'Z': 'dim_out'
    }
    _decay_params = ['weights']

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
        return params

    def dropout(x, act, rate):
        if act == 'T.tanh':
            x_d = self.trng.binomial(x.shape, p=1-rate, n=1,
                                     dtype=x.dtype)
            x = 2. * (x_d * (x + 1.) / 2) / (1 - rate) - 1
        elif act in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
            x_d = self.trng.binomial(x.shape, p=1-rate, n=1, dtype=x.dtype)
            x = x * x_d / (1 - rate)
        else:
            raise NotImplementedError('No dropout for %s yet' % activ)
        return x

    def feed(self, X, *params):
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
                    X = self.dropout(X, self.h_act, self.dropout)
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

    _required = ['dim_in', 'dim_out', 'distribution_type']
    _components = {
        'mlp': {
            'cell_type': 'MLP',
            '@required': {'out_dist': 'identity'},
            '@passed': ['dim_h', 'n_layers', 'dim_hs', 'h_act']
        },
        'distribution': {
            'cell_type': '&distribution_type',
            '@required': {'conditional': True}
        },
    }
    _links = [('mlp.output, dist.input')]

    def __init__(self, dim_in, dim_out, distribution_type, name=None, **kwargs):
        if name is None:
            name = '%s_%s' % (distribution_type, 'MLP')
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.distribution_type = distribution_type
        super(DistributionMLP, self).__init__(name=name, **kwargs)

    def set_components(self, **kwargs):
        from ..utils.tools import _p
        args = {}
        classes = {}

        for k, v in self._components.iteritems():
            name = _p(self.name, k)
            required = v.get('@required', [])
            passed = v.get('@passed', [])
            cell_args = kwargs.get(k, dict())

            if name in self.cell_manager.cells.keys():
                cell = self.cell_manager[name]
                C = cell.__class__
                components[k] = component
                for kk, vv in required.iteritems():
                    if cell.__dict__[kk] != vv:
                        raise TypeError(
                            'Provided cell %s is a components of %s. %s '
                            'should be %s but is %s'
                            % (name, self.name, cell.__dict__[kk], vv))
                for kk in passed:
                    vv = kwargs.pop(kk, None)
                    if vv is not None and cell.__dict__[kk] != vv:
                        raise TypeError(
                            'Provided cell %s is a components of %s. %s '
                            'should be %s but is %s'
                            % (name, self.name, cell.__dict__[kk], vv))

            else:
                if name in self.cell_manager.cell_args.keys():
                    cell_args.update(**self.cell_manager.cell_args[name])
                for kk, vv in required.iteritems():
                    cell_args[kk] = vv
                for kk in passed:
                    vv = kwargs.pop(kk, None)
                    if vv is not None:
                        cell_args[kk] = vv

                c = cell_args.get('cell_type', v.get('cell_type', None))
                if c.startswith('&'):
                    c = eval('self.' + c[1:])
                C = resolve_class(c)
                if C is None:
                    raise TypeError(
                        '`cell_type` of %s must be provided, either in defaults'
                        ' or in provided arguments.'
                    )
                args[k] = cell_args

            classes[k] = C

        print args
        print classes
        assert False

        for f, t in _links:
            s_f = f.split('.')
            f_name = '.'.join(s_f[:-1])
            f_arg = s_f[-1]

            s_t = t.split('.')
            t_name = '.'.join(s_t[:-1])
            t_arg = s_t[-1]

            f_arg = classes[f_name]._arg_map.get(f_arg, f_arg)
            t_arg = classes[t_name]._arg_map.get(t_arg, t_arg)


    def feed(self, X, *params):
        outs = self.mlp.feed(X, *params)
        Y = outs['Y']
        outs.update(**self.distribution(Y))
        return outs

    def sample(self, *args, **kwargs):
        assert self.distribution is not None
        return self.distribution.sample(*args, **kwargs)

    def neg_log_prob(self, *args, **kwargs):
        assert self.distribution is not None
        return self.distribution.neg_log_prob(*args, **kwargs)

    def entropy(self, *args, **kwargs):
        assert self.distribution is not None
        return self.distribution.entropy(*args, **kwargs)

    def get_center(self, *args, **kwargs):
        assert self.distribution is not None
        return self.distribution.get_center(*args, **kwargs)

    def split_prob(self, *args, **kwargs):
        assert self.distribution is not None
        return self.distribution.split_prob(*args, **kwargs)


_classes = {'MLP': MLP, 'DistributionMLP': DistributionMLP}