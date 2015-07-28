'''
Module for general layers
'''
import numpy as np
import theano
import tools

from collections import OrderedDict
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T


floatX = theano.config.floatX


class Layer(object):
    def __init__(self, name='', learn=True):
        self.name = name
        self.params = None
        self.excludes = []
        self.learn = learn

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


class FFN(Layer):
    def __init__(self, nin, nout, ortho=True, dropout=False, trng=None,
                 activ='lambda x: x', name='ff', weight_noise=False):
        self.nin = nin
        self.nout = nout
        self.ortho = ortho
        self.activ = activ
        self.dropout = dropout
        self.weight_noise = weight_noise

        if self.dropout and trng is None:
            self.trng = RandomStreams(1234)
        else:
            self.trng = trng

        super(FFN, self).__init__(name)
        self.set_params()

    def set_params(self):
        W = tools.norm_weight(self.nin, self.nout, scale=0.01, ortho=self.ortho)
        b = np.zeros((self.nout,)).astype(floatX)

        self.params = OrderedDict(
            W=W,
            b=b
        )

        if self.weight_noise:
            W_noise = (W * 0).astype(floatX)
            self.params.update(W_noise=W_noise)

    def __call__(self, state_below):
        W = self.W + self.W_noise if self.weight_noise else self.W
        z = eval(self.activ)(T.dot(state_below, W) + self.b)
        if self.dropout:
            if self.activ == 'T.tanh':
                raise NotImplementedError()
            else:
                z_d = self.trng.binomial(z.shape, p=1-self.dropout, n=1,
                                         dtype=z.dtype)
                z = z * z_d / (1 - self.dropout)
        return OrderedDict(z=z), theano.OrderedUpdates()


class MLP(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers, rng=None, trng=None,
                 weight_scale=0.1, weight_noise=False, dropout=False,
                 h_act='T.nnet.sigmoid', out_act='T.nnet.sigmoid',
                 name='MLP'):

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.n_layers = n_layers
        assert n_layers > 0

        self.weight_scale = weight_scale
        self.weight_noise = weight_noise
        self.dropout = dropout

        self.h_act = h_act
        self.out_act = out_act

        super(MLP, self).__init__(name=name)
        self.set_params()

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == self.n_layers - 1:
                dim = self.dim_out
            else:
                dim = self.dim_h

            W = tools.norm_weight(self.dim_in, dim,
                                  scale=self.weight_scale, ortho=False)
            b = np.zeros((dim,)).astype(floatX)

            self.params['W_%d' % l] = W
            self.params['b_%d' % l] = b

            if self.weight_noise:
                W_noise = (W * 0).astype(floatX)
                self.params['W_%d_noise' % l] = W_noise

    def __call__(self, x):
        for l in xrange(self.n_layers):
            W = self.__dict__['W_%d' % l]
            b = self.__dict__['b_%d' % l]

            if self.weight_noise:
                W_noise = self.__dict__['W_%d_noise' % l]
                W = W + W_noise

            if l == self.n_layers - 1:
                activ = self.out_act
            else:
                activ = self.h_act

            x = eval(activ)(T.dot(x, W) + b)

            if self.dropout:
                if activ == 'T.tanh':
                    raise NotImplementedError()
                else:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)

        return x

    def get_params(self):
        params = []
        for l in xrange(self.n_layers):
            W = self.__dict__['W_%d' % l]
            b = self.__dict__['b_%d' % l]
            params += [W, b]

            if self.weight_noise:
                W_noise = self.__dict__['W_%d_noise' % l]
                params += [W_noise]

        return params

    def step_call(self, x, *params):
        params = list(params)

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                W_noise = params.pop(0)
                W = W + W_noise

            if l == self.n_layers - 1:
                activ = self.out_act
            else:
                activ = self.h_act

            x = eval(activ)(T.dot(x, W) + b)

            if self.dropout:
                if activ == 'T.tanh':
                    raise NotImplementedError()
                else:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
        return x


class Softmax(Layer):
    def __init__(self, name='softmax'):
        super(Softmax, self).__init__(name)

    def __call__(self, input_):
        if input_.ndim == 3:
            reshape = input_.shape
            input_ = input_.reshape((input_.shape[0] * input_.shape[1], input_.shape[2]))
        else:
            reshape = False
        y_hat = T.nnet.softmax(input_)
        if reshape:
            y_hat = y_hat.reshape(reshape)
        return OrderedDict(y_hat=y_hat), theano.OrderedUpdates()

    def err(self, y_hat, Y):
        err = (Y * (1 - y_hat) + (1 - Y) * y_hat).mean()
        return err

    def zero_one_loss(self, y_hat, Y):
        y_hat = self.take_argmax(y_hat)
        y = self.take_argmax(Y)
        return T.mean(T.neq(y_hat, y))

    def take_argmax(self, var):
        """Takes argmax along 1st axis if tensor variable is matrix."""
        if var.ndim == 2:
            return T.argmax(var, axis=1)
        elif var.ndim == 1:
            return var
        else:
            raise ValueError("Un-recognized shape for Softmax::"
                             "zero-one-loss, taking argmax!")


class Logistic(Layer):
    def __init__(self, name='logistic'):
        super(Logistic, self).__init__(name)

    def __call__(self, input_):
        y_hat = T.nnet.sigmoid(input_)
        return OrderedDict(y_hat=y_hat), theano.OrderedUpdates()

    def err(self, y_hat, Y):
        err = (Y * (1 - y_hat) + (1 - Y) * y_hat).mean()
        return err


class Baseline(Layer):
    def __init__(self, name='baseline', rate=0.1):
        self.rate = np.float32(rate)
        super(Baseline, self).__init__(name)
        self.set_params()

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
