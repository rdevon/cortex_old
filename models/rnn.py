"""
Module for RNN layers
"""

import copy
from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from utils import tools
from utils.tools import (
    gaussian,
    log_gaussian,
    init_rngs,
    init_weights,
    norm_weight,
    ortho_weight
)


floatX = theano.config.floatX

def raise_type_error(o, t):
    raise ValueError('%s is not of type %s' % (type(o), t))

pi = theano.shared(np.pi).astype(floatX)

def init_h(h_init, X, batch_size, models, **h_args):
    if h_init is None:
        h0 = None
    elif h_init == 'last':
        print 'Initializing h0 from chain'
        h0 = theano.shared(np.zeros((batch_size, rnn.dim_h)).astype(floatX))
        h0s = h0[None, :, :]
    elif h_init == 'noise':
        noise_amount = h_args['noise_amount']
        print 'Initializing h0 from noise'
        h0 = trng.normal(avg=0, std=0.1, size=(batch_size, rnn.dim_h)).astype(floatX)
        h0s = h0[None, :, :]
    elif h_init == 'average':
        print 'Initializing h0 from running average'
        averager = models['averager']
        h0 = (T.alloc(0., batch_size, rnn.dim_h) + averager.m[None, :]).astype(floatX)
        h0s = h0[None, :, :]
    elif h_init == 'mlp':
        print 'Initializing h0 from MLP'
        mlp = models['h_net']
        h0s = mlp(X)

    return h0s

class RNN(Layer):
    def __init__(self, dim_in, dim_h,
                 conditional=None, input_net=None, output_net=None,
                 name='gen_rnn', **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.input_net = input_net
        self.output_net = output_net
        self.conditional = conditional

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)
        assert len(kwargs) == 0, kwargs.keys()
        super(RNN, self).__init__(name=name)

    def set_params(self):
        Ur = ortho_weight(self.dim_h)
        self.params = OrderedDict(Ur=Ur)
        self.set_net_params()

    def set_net_params(self):
        if self.input_net is None:
            self.input_net = MLP(
                self.dim_in, self.dim_h, self.dim_h, 1,
                rng=self.rng, trng=self.trng,
                h_act='T.nnet.sigmoid', out_act='T.tanh',
                name='input_net')
        else:
            assert self.input_net.dim_in == self.dim_in
            assert self.input_net.dim_out == self.dim_h
            self.input_net.name = 'input_net'

        if self.output_net is None:
            self.output_net = MLP(
                self.dim_h, self.dim_h, self.dim_in, 1,
                rng=self.rng, trng=self.trng,
                h_act='T.nnet.sigmoid', out_act='T.nnet.sigmoid',
                name='output_net')
        else:
            assert self.output_net.dim_in == self.dim_h
            self.output_net.name = 'output_net'

        if self.conditional is not None:
            assert self.conditional.dim_in == self.dim_in
            assert self.conditional.dim_out == self.dim_in
            self.conditional.name = 'conditional'
            self.nets.append(self.conditional)

        self.nets = [self.input_net, self.output_net, self.conditional]

    def set_tparams(self):
        tparams = super(RNN, self).set_tparams()
        for net in self.nets:
            if net is not None:
                tparams.update(**net.set_tparams())

        self.param_idx = [1]
        accum = self.param_idx[0]
        for net in self.nets:
            if net is not None:
                accum += len(net.get_params())
            self.param_idx.append(accum)

        return tparams

    def get_params(self):
        params = [self.Ur]
        return params

    def get_sample_params(self):
        params = [self.Ur]
        for net in self.nets:
            if net is not None:
                params += net.get_params()
        return params

    def get_recurrent_args(self, *args):
        return args[:self.param_idx[0]]

    def get_input_args(self, *args):
        return args[self.param_idx[0]:self.param_idx[1]]

    def get_output_args(self, *args):
        return args[self.param_idx[1]:self.param_idx[2]]

    def get_conditional_args(self, *args):
        return args[self.param_idx[2]:self.param_idx[3]]

    def step_sample(self, h_, x_, *params):
        Ur = self.get_recurrent_args(*params)[0]
        input_params = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y = self.input_net.preact(x_, *a_params)
        h = self._step(y, h_, Ur)

        preact = self.output_net.preact(h, *o_params)
        if self.conditional is not None:
            c_params = self.get_conditional_args(*params)
            preact += self.conditional.preact(x_, *c_params)

        p = eval(self.output_net.out_act)(preact)
        x = self.output_net.sample(p)
        return h, x, p

    def _step(self, y, h_, Ur):
        preact = T.dot(h_, Ur) + y
        h = T.tanh(preact)
        return h

    def energy(self, X, h0=None):
        outs, updates = self.__call__(X[:-1], h0=h0)
        p = outs['p']
        energy = self.neg_log_prob(X[1:], p).sum(axis=0)
        return energy

    def neg_log_prob(self, x, p):
        return self.output_net.neg_log_prob(x, p)

    def sample(self, x0=None, h0=None, n_samples=10, n_steps=10):
        if x0 is None:
            x0 = self.output_net.sample(
                p=0.5, size=(n_samples, self.output_net.dim_out)).astype(floatX)

        p0 = x0.copy()
        if h0 is None:
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX)

        seqs = []
        outputs_info = [h0, x0, None]
        non_seqs = self.get_sample_params()

        (h, x, p), updates = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_sampling'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        x = tools.concatenate([x0[None, :, :], x])
        h = tools.concatenate([h0[None, :, :], h])
        p = tools.concatenate([p0[None, :, :], p])

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0), updates

    def call_seqs(self, x):
        a = self.input_net(x, return_preact=True)
        seqs = [a]
        return seqs

    def __call__(self, x, h0=None):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        if h0 is None:
            h0 = T.alloc(0., n_samples, self.dim_h).astype(floatX)

        seqs = self.call_seqs(x)
        outputs_info = [h0]
        non_seqs = self.get_params()

        h, updates = theano.scan(
            self._step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=self.name + '_recurrent_steps',
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        preact = self.output_net(h, return_preact=True)
        if self.conditional is not None:
            preact += self.conditional(x, return_preact=True)
        p = eval(self.output_net.out_act)(preact)
        y = self.output_net.sample(p=p)

        return OrderedDict(h=h, y=y, p=p), updates
