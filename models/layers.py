'''
Module for general layers
'''

from collections import OrderedDict
import copy
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from utils import tools
from utils.tools import (
    concatenate,
    log_mean_exp,
    init_rngs,
    init_weights,
    _slice,
    _slice2
)


floatX = theano.config.floatX
pi = theano.shared(np.pi).astype('float32')
e = theano.shared(np.e).astype('float32')


class Layer(object):
    def __init__(self, name='', learn=True):
        self.name = name
        self.params = None
        self.excludes = []
        self.learn = learn
        self.set_params()

    def set_params(self):
        raise NotImplementedError()

    def set_tparams(self):
        if self.params is None:
            raise ValueError('Params not set yet')
        tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            tp = theano.shared(self.params[kk], name=kk)
            tparams[tools._p(self.name, kk)] = tp
            self.__dict__[kk] = tp
        return tparams

    def get_excludes(self):
        if self.learn:
            return [tools._p(self.name, e) for e in self.excludes]
        else:
            return [tools._p(self.name, k) for k in self.params.keys()]

    def __call__(self, state_below):
        raise NotImplementedError()


class ParzenEstimator(Layer):
    def __init__(self, dim, name='parzen'):
        self.dim = dim
        super(ParzenEstimator, self).__init__(name=name)

    def set_params(self, samples, x):
        sigma = self.get_sigma(samples, x).astype('float32')
        self.params = OrderedDict(sigma=sigma)

    def get_sigma(self, samples, xs):
        sigma = np.zeros((samples.shape[-1],))
        for x in xs:
            dx = (samples - x[None, :]) ** 2
            sigma += T.sqrt(dx / np.log(np.sqrt(2 * pi))).mean(axis=(0))
        return sigma / float(xs.shape[0])

    def __call__(self, samples, x):
        z = T.log(self.sigma * T.sqrt(2 * pi)).sum()
        d_s = (samples[:, None, :] - x[None, :, :]) / self.sigma[None, None, :]
        e = log_mean_exp((-.5 * d_s ** 2).sum(axis=2), axis=0)
        return (e - z).mean()


class Scheduler(Layer):
    '''
    Class for tensor constant scheduling.
    This can be used to schedule updates to tensor parameters, such as learning rates given
    the epoch.
    '''
    def __init__(self, rate=1, method='lambda x: 2 * x', name='scheduler'):
        self.rate = rate
        self.method = method
        super(Scheduler, self).__init__(name=name)

    def set_params(self):
        counter = 0
        self.params = OrderedDict(counter=counter, switch=switch)

    def __call__(self, x):
        counter = T.switch(T.ge(self.counter, self.rate), 0, self.counter + 1)
        switch = T.switch(T.ge(self.counter, 0), 1, 0)
        x = T.switch(switch, eval(method)(x), x)

        return OrderedDict(x=x), theano.OrderedUpdates([(self.counter, counter)])

# Bernoulli
def _binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

def _centered_binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return 2 * trng.binomial(p=0.5*(p+1), size=size, n=1, dtype=p.dtype) - 1.

def _cross_entropy(x, p, axis=None, scale=1.0):
    p = T.clip(p, 1e-7, 1.0 - 1e-7)
    energy = T.nnet.binary_crossentropy(p, x)
    if axis is None:
        axis = energy.ndim - 1
    energy = energy.sum(axis=axis)
    return (scale * energy).astype('float32')

def _binary_entropy(p, axis=None):
    p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
    entropy = T.nnet.binary_crossentropy(p_c, p)
    if axis is None:
        axis = entropy.ndim - 1
    entropy = entropy.sum(axis=axis)
    return entropy

# Softmax
def _softmax(x, axis=None):
    if axis is None:
        axis = x.ndim - 1
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def _sample_softmax(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.multinomial(pvals=p, size=size).astype('float32')

def _categorical_cross_entropy(x, p, axis=None, scale=1.0):
    p = T.clip(p, 1e-7, 1.0 - 1e-7)
    #energy = T.nnet.categorical_crossentropy(p, x)
    energy = T.nnet.binary_crossentropy(p, x)
    if axis is None:
        axis = x.ndim - 1
    energy = energy.sum(axis=axis)
    return (scale * energy / p.shape[p.ndim-1]).astype('float32')

def _categorical_entropy(p, axis=None):
    p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
    entropy = T.nnet.categorical_crossentropy(p_c, p)
    return entropy

# Gaussian
def _normal(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)

    if size is None:
        size = mu.shape
    else:
        pass
#        mu = T.zeros(size) + mu[None, :, :]
#        log_sigma = T.zeros(size) + log_sigma[None, :, :]

    #return mu + T.exp(log_sigma) * trng.normal(avg=0.0, std=1.0, size=size)
    return trng.normal(avg=mu, std=T.exp(log_sigma), size=size)

def _normal_prob(p):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    return mu

def _neg_normal_log_prob(x, p, axis=None, scale=1.0):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)
    energy = 0.5 * (
        (x - mu)**2 / (T.exp(2 * log_sigma)) + 2 * log_sigma + T.log(2 * pi))

    if axis is None:
        axis = energy.ndim - 1
    energy = energy.sum(axis=axis)
    return (scale * energy).astype('float32')

def _normal_entropy(p, axis=None):
    dim = p.shape[p.ndim-1] // 2
    log_sigma = _slice(p, 1, dim)

    entropy = 0.5 * T.log(2 * pi * e) + log_sigma
    if axis is None:
        axis = entropy.ndim - 1
    entropy = entropy.sum(axis=axis)
    return entropy


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

            W = tools.norm_weight(dim_in, dim_out,
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
        neg_log_prob = T.constant(0.).astype('float32')
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
        entropy = T.constant(0.).astype('float32')
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


class Averager(Layer):
    def __init__(self, shape, name='averager', rate=0.1):
        self.rate = np.float32(rate)
        self.shape = shape
        super(Averager, self).__init__(name)

    def set_params(self):
        m = np.zeros(self.shape).astype(floatX)
        self.params = OrderedDict(m=m)

    def __call__(self, x):
        if x.ndim == 1:
            m = x
        elif x.ndim == 2:
            m = x.mean(axis=0)
        elif x.ndim == 3:
            m = x.mean(axis=(0, 1))
        else:
            raise ValueError()

        new_m = ((1. - self.rate) * self.m + self.rate * m).astype(floatX)
        updates = [(self.m, new_m)]
        return OrderedDict(m=new_m), updates


class Baseline(Layer):
    def __init__(self, name='baseline', rate=0.1):
        self.rate = np.float32(rate)
        super(Baseline, self).__init__(name)

    def set_params(self):
        m = np.float32(0.)
        var = np.float32(0.)

        self.params = OrderedDict(m=m, var=var)

    def __call__(self, input_):
        m = input_.mean()
        v = input_.std()

        new_m = T.switch(T.eq(self.m, 0.),
                         m,
                         (np.float32(1.) - self.rate) * self.m + self.rate * m)
        new_var = T.switch(T.eq(self.var, 0.),
                           v,
                           (np.float32(1.) - self.rate) * self.var + self.rate * v)

        updates = [(self.m, new_m), (self.var, new_var)]

        input_centered = (
            (input_ - new_m) / T.maximum(1., T.sqrt(new_var)))

        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x=input_,
            x_centered=input_centered,
            m=new_m,
            var=new_var
        )
        return outs, updates

class BaselineWithInput(Baseline):
    def __init__(self, dims_in, dim_out, rate=0.1, name='baseline_with_input'):
        if len(dims_in) < 1:
            raise ValueError('One or more dims_in needed, %d provided'
                             % len(dims_in))
        self.dims_in = dims_in
        self.dim_out = dim_out
        super(BaselineWithInput, self).__init__(name=name, rate=rate)

    def set_params(self):
        super(BaselineWithInput, self).set_params()
        for i, dim_in in enumerate(self.dims_in):
            w = np.zeros((dim_in, self.dim_out)).astype('float32')
            k = 'w%d' % i
            self.params[k] = w

    def __call__(self, input_, update_params, *xs):
        '''
        Maybe unclear: input_ is the variable to be baselined, xs are the
        actual inputs.
        '''
        m = input_.mean()
        v = input_.std()

        new_m = T.switch(T.eq(self.m, 0.), m + self.m,
                         (np.float32(1.) - self.rate) * self.m + self.rate * m)
        new_m.name = 'new_m'
        new_var = T.switch(T.eq(self.var, 0.), v + self.var,
                           (np.float32(1.) - self.rate) * self.var + self.rate * v)

        if update_params:
            updates = [(self.m, new_m), (self.var, new_var)]
        else:
            updates = theano.OrderedUpdates()

        if len(xs) != len(self.dims_in):
            raise ValueError('Number of (external) inputs for baseline must'
                             ' match parameters')

        ws = []
        for i in xrange(len(xs)):
            # Maybe not the most pythonic way...
            ws.append(self.__dict__['w%d' % i])

        idb = T.sum([x.dot(W) for x, W in zip(xs, ws)], axis=0).T
        idb_c = T.zeros_like(idb) + idb
        input_centered = (
            (input_ - idb_c - new_m) / T.maximum(1., T.sqrt(new_var)))
        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x_c=input_,
            x_centered=input_centered,
            m=new_m,
            var=new_var,
            idb=idb,
            idb_c=idb_c
        )

        return outs, updates

class ScalingWithInput(Layer):
    def __init__(self, dims_in, dim_out, name='scaling_with_input'):
        if len(dims_in) < 1:
            raise ValueError('One or more dims_in needed, %d provided'
                             % len(dims_in))
        self.dims_in = dims_in
        self.dim_out = dim_out
        super(ScalingWithInput, self).__init__(name=name)
        self.set_params()

    def set_params(self):
        for i, dim_in in enumerate(self.dims_in):
            w = np.zeros((dim_in, self.dim_out)).astype('float32')
            k = 'w%d' % i
            self.params[k] = w

    def __call__(self, input_, *xs):
        '''
        Maybe unclear: input_ is the variable to be scaled, xs are the
        actual inputs.
        '''
        updates = theano.OrderedUpdates()

        if len(xs) != len(self.dims_in):
            raise ValueError('Number of (external) inputs for baseline must'
                             ' match parameters')

        ws = []
        for i in xrange(len(xs)):
            # Maybe not the most pythonic way...
            ws.append(self.__dict__['w%d' % i])

        ids = T.sum([x.dot(W) for x, W in zip(xs, ws)], axis=0).T
        ids_c = T.zeros_like(ids) + ids
        input_scaled = input_ / ids_c
        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x_c=input_,
            x_scaled=input_scaled,
            ids=ids,
            ids_c=ids_c
        )

        return outs, updates
