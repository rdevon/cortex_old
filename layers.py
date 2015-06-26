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


class Softmax(Layer):
    def __init__(self, name='softmax'):
        super(Softmax, self).__init__(name)

    def __call__(self, input_):
        y_hat = T.nnet.softmax(input_)
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
        c = np.float32(0.)
        var = np.float32(0.)

        self.params = OrderedDict(c=c, var=var)
        self.excludes=['c', 'var']

    def __call__(self, input_):
        c = input_.mean()
        v = input_.std()

        new_c = T.switch(T.eq(self.c, 0.),
                         c,
                         (np.float32(1.) - self.rate) * self.c + self.rate * c)
        new_var = T.switch(T.eq(self.var, 0.),
                           v,
                           (np.float32(1.) - self.rate) * self.var + self.rate * v)

        updates = [(self.c, new_c), (self.var, new_var)]

        input_centered = (
            (input_ - new_c) / T.maximum(1., T.sqrt(new_var)))

        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x=input_,
            x_centered=input_centered,
            c=new_c,
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
        c = input_.mean()
        v = input_.std()

        new_c = T.switch(T.eq(self.c, 0.),
                         c,
                         (np.float32(1.) - self.rate) * self.c + self.rate * c)
        new_c.name = 'new_c'
        new_var = T.switch(T.eq(self.var, 0.),
                           v,
                           (np.float32(1.) - self.rate) * self.var + self.rate * v)

        if update_params:
            updates = [(self.c, new_c), (self.var, new_var)]
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
        idb.name = 'idb'
        input_ = T.zeros_like(input_) + input_
        input_centered = (
            (input_ - idb - new_c) / T.maximum(1., T.sqrt(new_var)))

        outs = OrderedDict(
            x=input_,
            x_centered=input_centered,
            c=new_c,
            var=new_var,
            idb=idb
        )

        return outs, updates
