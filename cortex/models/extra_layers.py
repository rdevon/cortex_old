'''
Module for general layers
'''

from collections import OrderedDict
import copy
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from . import Cell
from . import mlp as mlp_module
from .. import utils
from ..utils import e, floatX, pi
from ..utils.maths import log_mean_exp


# SOME MISC LAYERS ------------------------------------------------------------


class Averager(Cell):
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
        return updates


class Baseline(Cell):
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

class ScalingWithInput(Cell):
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


class Attention(Cell):
    _components = ['mlp']

    def __init__(self, dim_in, dim_out, mlp=None, name='attention',
                 mlp_args=None, **kwargs):
        if mlp is None:
            if mlp_args is None:
                mlp_args = dict()
            mlp = mlp_module.factory(dim_in=dim_in, dim_out=dim_out,
                              name=name + '_mlp',
                              distribution='centered_binomial',
                              **mlp_args)
        self.mlp = mlp
        self.dim_out = dim_out
        kwargs = init_rngs(self, **kwargs)
        super(Attention, self).__init__(name=name, **kwargs)

    def set_params(self):
        v = self.rng.normal(size=(self.dim_out,)).astype(floatX)
        self.params = OrderedDict(v=v)

    def set_tparams(self):
        tparams = super(Attention, self).set_tparams()
        tparams.update(**self.mlp.set_tparams())
        return tparams

    def __call__(self, X, axis=0):
        X = utils.concatenate(X, axis=X.ndim-1)
        Y = self.mlp.feed(X)
        a = T.dot(Y, self.v)
        e = a / a.sum(axis=axis)
        return OrderedDict(a=a, e=e)


class Attention2(Cell):
    _components = ['mlp']

    def __init__(self, dim_in, dim_out, mlp=None, name='attention',
                 **kwargs):
        if mlp is None:
            mlp = mlp_module.factory(dim_in=dim_in, dim_out=dim_out,
                              name=name + '_mlp',
                              distribution='centered_binomial')
        self.mlp = mlp
        self.dim_out = dim_out
        kwargs = init_rngs(self, **kwargs)
        super(Attention, self).__init__(name=name, **kwargs)

    def set_params(self):
        v = self.rng.normal(size=(self.dim_out,)).astype(floatX)
        self.params = OrderedDict(v=v)

    def set_tparams(self):
        tparams = super(Attention, self).set_tparams()
        tparams.update(**self.mlp.set_tparams())
        return tparams

    def __call__(self, Xa, Xb, axis=0):
        X = utils.concatenate([Xa, Xb], axis=Xa.ndim-1)
        Y = self.mlp.feed(X)
        a = T.dot(Y, self.v)
        e = a / a.sum(axis=axis)
        return OrderedDict(a=a, e=e)


_classes = {'Averager': Averager}