'''
Module for MLP model.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
import warnings

from .distributions import Distribution, resolve as resolve_distribution
from . import Layer
from ..utils import floatX
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    norm_weight
)


def resolve(c):
    '''Resolves the MLP subclass from str.'''
    if c == 'mlp' or c is None:
        return MLP
    elif c == 'lfmlp':
        return LFMLP
    elif c == 'mmmlp':
        return MultimodalMLP
    else:
        raise ValueError(c)


class MLP(Layer):
    '''Multilayer perceptron model.

    Attributes:
        dim_in: int, input dimension.
        dim_out: int, output dimension.
        distribution: Distribution (optional), distribution of output. Used for
            sampling, density calculations, etc.
        dim_h: int (optional): dimention of hidden layer.
        dim_hs: list of ints (optional), for multiple hidden layers.
        n_layers: int, number of output and hidden layers.
    '''
    must_sample = False
    def __init__(self, dim_in, dim_out, dim_h=None, n_layers=None, dim_hs=None,
                 f_sample=None, f_neg_log_prob=None, f_entropy=None,
                 h_act='T.nnet.sigmoid', distribution='binomial', out_act=None,
                 distribution_args=dict(),
                 name='MLP',
                 **kwargs):

        self.dim_in = dim_in

        if out_act is not None:
            warnings.warn('out_act option going away. Use `distribution`.', FutureWarning)
            if out_act == 'T.nnet.sigmoid':
                distribution = 'binomial'
            elif out_act == 'T.tanh':
                distribution = 'centered_binomial'
            elif out_act == 'T.nnet.softmax':
                distribution = 'multimnomial'
            elif out_act == 'lambda x: x':
                distribution = 'gaussian'
            elif out_act == 'T.tanh':
                distribution = 'centered binomial'
            else:
                raise ValueError(out_act)

        if isinstance(distribution, Distribution):
            self.distribution = distribution
        elif distribution is not None:
            self.distribution = resolve_distribution(
                distribution, conditional=True)(
                dim_out, **distribution_args)
        else:
            self.distribution = None

        if self.distribution is not None:
            self.dim_out = dim_out * self.distribution.scale

        if dim_h is None:
            if dim_hs is None:
                dim_hs = []
            else:
                dim_hs = [dim_h for dim_h in dim_hs]
            assert n_layers is None
        else:
            assert dim_hs is None
            dim_hs = []
            for l in xrange(n_layers - 1):
                dim_hs.append(dim_h)
        dim_hs.append(self.dim_out)
        self.dim_hs = dim_hs
        self.n_layers = len(dim_hs)
        assert self.n_layers > 0

        self.h_act = h_act

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)
        super(MLP, self).__init__(name=name, **kwargs)

    @staticmethod
    def factory(dim_in=None, dim_out=None,
                **kwargs):
        return MLP(dim_in, dim_out, **kwargs)

    def l2_decay(self, gamma, layers=None):
        if layers is None:
            layers = range(self.n_layers)

        cost = T.constant(0.).astype(floatX)
        for l in layers:
            W = self.__dict__['W%d' % l]
            cost += gamma * (W ** 2).sum()

        return cost

    def sample(self, p, n_samples=1):
        assert self.distribution is not None
        return self.distribution.sample(p=p, n_samples=n_samples)

    def neg_log_prob(self, x, p, sum_probs=True):
        assert self.distribution is not None
        return self.distribution.neg_log_prob(x, p, sum_probs=sum_probs)

    def entropy(self, p):
        assert self.distribution is not None
        return self.distribution.entropy(p)

    def get_center(self, p):
        assert self.distribution is not None
        return self.distribution.get_center(p)

    def split_prob(self, p):
        assert self.distribution is not None
        return self.distribution.split_prob(p)

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_hs[l-1]
            dim_out = self.dim_hs[l]

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

    def step_call(self, x, *params):
        params = list(params)
        outs = OrderedDict(x=x)
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                print 'Using weight noise in layer %d for MLP %s' % (l, self.name)
                W += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)

            preact = T.dot(x, W) + b

            if l < self.n_layers - 1:
                x = eval(self.h_act)(preact)
                outs['preact_%d' % l] = preact
                outs[l] = x
            else:
                if self.distribution is not None:
                    x = self.distribution(preact)
                else:
                    x = eval(self.h_act)(preact)
                outs['z'] = preact
                outs['p'] = x

            if self.dropout and l != self.n_layers - 1:
                print 'Adding dropout to layer {layer} for MLP "{name}"'.format(
                    layer=l, name=self.name)
                if self.h_act == 'T.tanh':
                    raise NotImplementedError('dropout for tanh units not implemented yet')
                elif self.h_act in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
                else:
                    raise NotImplementedError('No dropout for %s yet' % activ)

        assert len(params) == 0, params
        return outs

    def __call__(self, x):
        params = self.get_params()
        outs = self.step_call(x, *params)
        return outs

    def feed(self, x):
        return self.__call__(x)['p']

    def step_feed(self, x, *params):
        return self.step_call(x, *params)['p']

    def preact(self, x):
        return self.__call__(x)['z']

    def step_preact(self, x, *params):
        return self.step_call(x, *params)['z']


class LFMLP(MLP):
    '''
    Local filters MLP (In progress)
    '''
    def __init__(self, dim_in, dim_out, dim_h=None, dim_hs=None, n_layers=None,
                 dim_f=None, filter_in=True, prototype=None, stride=1, shape=None,
                 name='LFMLP', **kwargs):

        self.filter_in = filter_in

        if prototype is None:
            assert isinstance(shape, list)
            if self.filter_in:
                assert reduce(lambda x, y: x + y, shape) == dim_in
            else:
                assert reduce(lambda x, y: x + y, shape) == dim_out
            accum = 0
            filter_idx = []
            for s in shape:
                filter_idx.append(range(accum, accum + s))
                accum += s
        else:
            if self.filter_in:
                assert prototype.sum() == dim_in
            else:
                assert prototype.sum() == dim_out

            assert len(prototype.shape) == len(shape)

            def make_filter_idx(f_shape, shape, start):
                idx = range(start[0], f_shape[0] + start[0])
                for st, f_s, s in zip(start[:-1][::-1], f_shape[:-1][::-1], shape[:-1][::-1]):
                    idx_new = []
                    for i in idx:
                        idx_new += [s * i + j for j in range(st, f_s + st)]
                    idx = idx_new
                return idx

            incs = [range(0, s - f_s + 1, stride) for s, f_s in zip(prototype.shape, shape)]
            starts = [[i] for i in incs[0]]
            for inc in incs[1:]:
                new_starts = []
                for i in inc:
                    new_starts += [s + [i] for s in starts]
                starts = new_starts
            filter_idx = [make_filter_idx(shape, prototype.shape, start) for start in starts]

            mask_idx = [i for i, j in enumerate(prototype.flatten()) if j == 1]
            idx_mask = dict((j, i) for i, j in enumerate(mask_idx))

            filter_idx = [[idx_mask[i] for i in f_idx if i in mask_idx] for f_idx in filter_idx]

        print 'Formed %d filters' % len(filter_idx)

        self.filter_idx = filter_idx

        assert dim_f is not None

        if dim_h is None:
            if dim_hs is None:
                dim_hs = []
            else:
                dim_hs = [dim_h for dim_h in dim_hs]
            assert n_layers is None
        else:
            assert dim_hs is None
            dim_hs = []
            for l in xrange(n_layers - 1):
                dim_hs.append(dim_h)

        if self.filter_in:
            dim_hs = [dim_f] + dim_hs
        else:
            dim_hs.append(dim_f)

        super(LFMLP, self).__init__(dim_in, dim_out, name=name, excludes=['f'],
                                    dim_h=None, n_layers=None, dim_hs=dim_hs,
                                    **kwargs)

    @staticmethod
    def factory(dim_in=None, dim_out=None,
                **kwargs):
        return LFMLP(dim_in, dim_out, **kwargs)

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_hs[l-1]
            dim_out = self.dim_hs[l]

            if self.filter_in:
                if l == 0:
                    f = np.zeros((len(self.filter_idx), dim_in)).astype(floatX)

                    for i, f_idx in enumerate(self.filter_idx):
                        f[i, f_idx] = 1

                    dim_out *= len(self.filter_idx)
                    self.params['f'] = f
                elif l == 1:
                    dim_in *= len(self.filter_idx)
            elif not self.filter_in:
                if l == self.n_layers - 1:
                    f = np.zeros((len(self.filter_idx), dim_out)).astype(floatX)
                    for i, f_idx in enumerate(self.filter_idx):
                        f[i, f_idx] = 1

                    dim_in *= len(self.filter_idx)
                    self.params['f'] = f
                elif l == self.n_layers - 2:
                    dim_out *= len(self.filter_idx)

            W = norm_weight(dim_in, dim_out,
                            scale=self.weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)
            self.params['W%d' % l] = W
            self.params['b%d' % l] = b

    def get_params(self):
        params = [self.f] + super(LFMLP, self).get_params()
        return params

    def step_call(self, x, f, *params):
        params = list(params)
        outs = OrderedDict(x=x)
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                print 'Using weight noise in layer %d for MLP %s' % (l, self.name)
                W += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)

            if ((self.filter_in and l == 0)
                or (not(self.filter_in) and l == self.n_layers - 1)):
                W_ = W[None, :, :] * f[:, :, None]
                W_ = W_.reshape((W_.shape[0] * W_.shape[1], W_.shape[2])).astype(floatX)
                preact = T.dot(x, W) + b
            else:
                preact = T.dot(x, W) + b

            if l < self.n_layers - 1:
                x = eval(self.h_act)(preact)
                outs['preact_%d' % l] = preact
                outs[l] = x
            else:
                if self.distribution is not None:
                    x = self.distribution(preact)
                else:
                    x = eval(self.h_act)(preact)
                outs['z'] = preact
                outs['p'] = x

            if self.dropout and l != self.n_layers - 1:
                print 'Adding dropout to layer {layer} for MLP {name}'.format(
                    layer=l, name=self.name)
                if activ == 'T.tanh':
                    raise NotImplementedError('dropout for tanh units not implemented yet')
                elif activ in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
                else:
                    raise NotImplementedError('No dropout for %s yet' % activ)

        assert len(params) == 0, params
        return outs


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

    def l2_decay(self, gamma, layers=None):
        if layers is None:
            layers = self.layers.keys()
            layers = [l for l in layers if l != 'i']

        cost = T.constant(0.).astype(floatX)
        for k in layers:
            W = self.__dict__['W_%s' % k]
            cost += gamma * (W ** 2).sum()
        
        rval = OrderedDict(cost = cost)
        return rval

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
