'''
Module for MLP model.
'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from distributions import (
    Bernoulli,
    Gaussian,
    _binomial,
    _cross_entropy,
    _binary_entropy,
    _centered_binomial,
    _normal,
    _neg_normal_log_prob,
    _normal_entropy,
    _normal_prob
)
from layers import Layer
from utils.tools import (
    concatenate,
    floatX,
    init_rngs,
    init_weights,
    norm_weight
)


class MLP(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers,
                 f_sample=None, f_neg_log_prob=None, f_entropy=None,
                 h_act='T.nnet.sigmoid', out_act='T.nnet.sigmoid',
                 name='MLP',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.n_layers = n_layers
        assert n_layers > 0

        self.h_act = h_act
        self.out_act = out_act

        if out_act is None:
            out_act = 'T.nnet.sigmoid'

        if out_act == 'T.nnet.sigmoid':
            self.f_sample = _binomial
            self.f_neg_log_prob = _cross_entropy
            self.f_entropy = _binary_entropy
            self.f_prob = lambda x: x
        elif out_act == 'T.tanh':
            self.f_sample = _centered_binomial
        elif out_act == 'lambda x: x':
            self.f_sample = _normal
            self.f_neg_log_prob = _neg_normal_log_prob
            self.f_entropy = _normal_entropy
            self.f_prob = _normal_prob
            self.dim_out *= 2
        else:
            assert f_sample is not None
            assert f_neg_log_prob is not None
            assert out_act is not None

        if f_sample is not None:
            self.sample = f_sample
        if f_neg_log_prob is not None:
            self.net_log_prob = f_neg_log_prob
        if f_entropy is not None:
            self.entropy = f_entropy

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        #assert len(kwargs) == 0, kwargs.keys()
        super(MLP, self).__init__(name=name)

    def sample(self, p, size=None):
        return self.f_sample(self.trng, p, size=size)

    def neg_log_prob(self, x, p):
        return self.f_neg_log_prob(x, p)

    def entropy(self, p):
        return self.f_entropy(p)

    def prob(self, p):
        return self.f_prob(p)

    @staticmethod
    def factory(dim_in=None, dim_h=None, dim_out=None, n_layers=None,
                **kwargs):
        return MLP(dim_in, dim_h, dim_out, n_layers, **kwargs)

    def get_L2_weight_cost(self, gamma, layers=None):
        if layers is None:
            layers = range(self.n_layers)

        cost = T.constant(0.).astype(floatX)
        for l in layers:
            W = self.__dict__['W%d' % l]
            cost += gamma * (W ** 2).sum()

        return cost

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h

            if l == self.n_layers - 1:
                dim_out = self.dim_out
            else:
                dim_out = self.dim_h

            W = norm_weight(dim_in, dim_out,
                                  scale=self.weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)

            self.params['W%d' % l] = W
            self.params['b%d' % l] = b

    def get_params(self):
        params = []
        for l in xrange(self.n_layers):
            W = self.__dict__['W%d' % l]
            b = self.__dict__['b%d' % l]
            params += [W, b]
        return params

    def preact(self, x, *params):
        # Used within scan with `get_params`
        params = list(params)

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                print 'Using weight noise in layer %d for MLP %s' % (l, self.name)
                W += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)

            if l == self.n_layers - 1:
                x = T.dot(x, W) + b
            else:
                activ = self.h_act
                x = eval(activ)(T.dot(x, W) + b)

            if self.dropout:
                if activ == 'T.tanh':
                    raise NotImplementedError('dropout for tanh units not implemented yet')
                elif activ in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
                else:
                    raise NotImplementedError('No dropout for %s yet' % activ)

        assert len(params) == 0, params
        return x

    def step_call(self, x, *params):
        x = self.preact(x, *params)
        return eval(self.out_act)(x)

    def __call__(self, x, return_preact=False):
        params = self.get_params()
        if return_preact:
            x = self.preact(x, *params)
        else:
            x = self.step_call(x, *params)

        return x

# MULTIMODAL MLP CLASS --------------------------------------------------------

class MultiModalMLP(Layer):
    def __init__(self, dim_in, graph, log_prob_scale=dict(), name='MLP',
                 **kwargs):
        graph = copy.deepcopy(graph)

        self.layers = OrderedDict()
        self.layers.update(**graph['layers'])
        self.edges = graph['edges']
        outs = graph['outs'].keys()
        for k in outs:
            assert not k in self.layers.keys()
        self.layers.update(**graph['outs'])

        for l in self.layers.keys():
            if self.layers[l]['act'] == 'lambda x: x':
                self.layers[l]['dim'] *= 2

        self.outs = OrderedDict()
        for i, o in self.edges:
            if o in outs:
                assert not o in self.outs.keys()
                o_dict = OrderedDict()
                act = self.layers[o]['act']
                if act == 'T.nnet.sigmoid':
                    o_dict['f_sample'] = _binomial
                    o_dict['f_neg_log_prob'] = _cross_entropy
                    o_dict['f_entropy'] = _binary_entropy
                    o_dict['f_prob'] = lambda x: x
                elif act == 'T.nnet.softmax':
                    o_dict['f_sample'] = _sample_softmax
                    o_dict['f_neg_log_prob'] = _categorical_cross_entropy
                    o_dict['f_entropy'] = _categorical_entropy
                    o_dict['f_prob'] = lambda x: x
                    self.layers[o]['act'] = '_softmax'
                elif act == 'T.tanh':
                    o_dict['f_sample'] = _centered_binomial
                elif act == 'lambda x: x':
                    o_dict['f_sample'] = _normal
                    o_dict['f_neg_log_prob'] = _neg_normal_log_prob
                    o_dict['f_entropy'] = _normal_entropy
                    o_dict['f_prob'] = _normal_prob
                else:
                    raise ValueError(act)

                if log_prob_scale.get(o, None) is not None:
                    o_dict['log_prob_scale'] = log_prob_scale[o]

                self.outs[o] = o_dict

        assert not 'i' in self.layers.keys()
        self.layers['i'] = dict(dim=dim_in)

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        #assert len(kwargs) == 0, 'Got extra args: %r' % kwargs.keys()
        super(MultiModalMLP, self).__init__(name=name)

    @staticmethod
    def factory(dim_in=None, graph=None, **kwargs):
        return MultiModalMLP(dim_in, graph, **kwargs)

    def sample(self, p, size=None, split=False):
        if size is None:
            size = p.shape
        start = 0
        x = []
        for o, v in self.outs.iteritems():
            dim = self.layers[o]['dim']
            f_sample = v['f_sample']
            p_ = _slice2(p, start, start+dim)
            if self.layers[o]['act'] == 'lambda x: x':
                scale = 2
            else:
                scale = 1

            if size is None:
                size_ = None
            else:
                if p.ndim == 1:
                    size_ = (size[0], p_.shape[0] // scale)
                elif p.ndim == 2:
                    size_ = (size[0], p_.shape[0], p_.shape[1] // scale)
                elif p.ndim == 3:
                    size_ = (size[0], p_.shape[0], p_.shape[1], p_.shape[2] // scale)
                else:
                    raise ValueError()
            x.append(f_sample(self.trng, p_, size=size_))
            start += dim

        if split:
            return x
        else:
            return concatenate(x, axis=(x[0].ndim-1))

    def neg_log_prob(self, x, p):
        neg_log_prob = T.constant(0.).astype(floatX)
        start = 0
        start_x = 0
        for o, v in self.outs.iteritems():
            dim = self.layers[o]['dim']
            f_neg_log_prob = v['f_neg_log_prob']
            log_prob_scale = v.get('log_prob_scale', 1.0)
            if self.layers[o]['act'] == 'lambda x: x':
                scale = 2
            else:
                scale = 1
            p_ = _slice2(p, start, start + dim)
            x_ = _slice2(x, start_x, start_x + dim // scale)
            neg_log_prob += f_neg_log_prob(x_, p_, scale=log_prob_scale)
            start += dim
            start_x += dim // scale

        return neg_log_prob

    def entropy(self, p):
        start = 0
        entropy = T.constant(0.).astype(floatX)
        for o, v in self.outs.iteritems():
            dim = self.layers[o]['dim']
            f_entropy = v['f_entropy']
            p_ = _slice2(p, start, start + dim)
            entropy += f_entropy(p_)
            start += dim

        return entropy

    def prob(self, p):
        start = 0
        x = []
        for o, v in self.outs.iteritems():
            dim = self.layers[o]['dim']
            f_prob = v['f_prob']
            p_ = _slice2(p, start, start + dim)
            x.append(f_prob(p_))
            start += dim

        return x

    def get_L2_weight_cost(self, gamma, layers=None):
        if layers is None:
            layers = self.layers.keys()
            layers = [l for l in layers if l != 'i']

        cost = T.constant(0.).astype(floatX)
        for k in layers:
            W = self.__dict__['W_%s' % k]
            cost += gamma * (W ** 2).sum()

        return cost

    def split(self, p):
        start = 0
        ps = []
        for o, v in self.outs.iteritems():
            dim = self.layers[o]['dim']
            p_ = _slice2(p, start, start + dim)
            ps.append(p_)
            start += dim

        return ps

    def set_params(self):
        self.params = OrderedDict()

        for i, o in self.edges:
            assert not o == 'i'
            assert not i in self.outs

            dim_in = self.layers[i]['dim']
            dim_out = self.layers[o]['dim']

            W = tools.norm_weight(dim_in, dim_out,
                                  scale=self.weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)

            self.params['W_%s' % o] = W
            self.params['b_%s' % o] = b

    def get_params(self):
        params = []
        for _, o in self.edges:
            W = self.__dict__['W_%s' % o]
            b = self.__dict__['b_%s' % o]
            params += [W, b]

        return params

    def preact(self, x, *params):
        # Used within scan with `get_params`
        params = list(params)
        outs = dict(i=x)

        for i, o in self.edges:
            x = outs[i]
            assert not o in outs.keys()
            W = params.pop(0)
            b = params.pop(0)

            if o in self.outs:
                x = T.dot(x, W) + b
            else:
                act = self.layers[o]['act']
                x = eval(act)(T.dot(x, W) + b)

            outs[o] = x

        x = []
        for o in self.outs.keys():
            x.append(outs[o])

        return concatenate(x, axis=(x[0].ndim-1))

    def step_call(self, x, *params):
        x = self.preact(x, *params)
        start = 0
        y = []
        for o in self.outs.keys():
            dim = self.layers[o]['dim']
            act = self.layers[o]['act']
            z_ = _slice2(x, start, start + dim)
            x_  = eval(act)(z_)
            y.append(x_)
            start += dim
        return concatenate(y, axis=(y[0].ndim-1))

    def __call__(self, x, return_preact=False):
        params = self.get_params()
        if return_preact:
            x = self.preact(x, *params)
        else:
            x = self.step_call(x, *params)
        return x