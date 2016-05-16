"""
Module for LSTM layers
"""

import copy
from collections import OrderedDict
import numpy as np
import random
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import Layer
from .mlp import MLP
from .rnn import RNN
from ..utils import floatX
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    norm_weight,
    ortho_weight,
    pi,
    scan,
    _slice
)

class LSTM(RNN):
    def __init__(self, dim_in, dim_h, name='lstm', **kwargs):

        super(LSTM, self).__init__(dim_in, dim_h, name=name, **kwargs)

    @staticmethod
    def factory(dim_in=None, dim_h=None, **kwargs):
        return RNN(dim_in, dim_h, **kwargs)

    @staticmethod
    def mlp_factory(dim_in, dim_h, data_iter, dim_out=None,
                    i_net=None, a_net=None, o_net=None, c_net=None):
        mlps = {}

        if dim_out is None:
            dim_out = dim_in

        if i_net is not None:
            i_net['distribution'] = 'centered_binomial'
            input_net = MLP.factory(dim_in=dim_in, dim_out=4*dim_h,
                                    name='input_net', **i_net)
            mlps['input_net'] = input_net

        if o_net is not None:
            o_net['distribution'] = data_iter.distributions[data_iter.name]
            output_net = MLP.factory(dim_in=dim_h, dim_out=dim_out,
                                     name='output_net', **o_net)
            mlps['output_net'] = output_net

        if c_net is not None:
            if not c_net.get('dim_in', False):
                c_net['dim_in'] = dim_in
            conditional = MLP.factory(dim_out=dim_h,
                                      name='conditional', **c_net)
            mlps['conditional'] = conditional

        return mlps

    def set_params(self):
        Ur = norm_weight(self.dim_h, 4 * self.dim_h,
                         scale=self.weight_scale, ortho=False)
        self.params = OrderedDict(Ur=Ur)
        self.set_net_params()

    def get_gates(self, x):
        i = T.nnet.sigmoid(_slice(x, 0, self.dim_h))
        f = T.nnet.sigmoid(_slice(x, 1, self.dim_h))
        o = T.nnet.sigmoid(_slice(x, 2, self.dim_h))
        c = T.tanh(_slice(x, 3, self.dim_h))
        return i, f, o, c

    def set_net_params(self):
        if self.input_net is None:
            self.input_net = MLP(
                self.dim_in, self.dim_h,
                rng=self.rng, trng=self.trng,
                distribution='centered_binomial',
                name='input_net')
        else:
            assert self.input_net.dim_in == self.dim_in
            assert self.input_net.dim_out == 4 * self.dim_h
        self.input_net.name = self.name + '_input_net'

        if self.output_net is None:
            self.output_net = MLP(
                self.dim_h, self.dim_out,
                rng=self.rng, trng=self.trng,
                distribution='binomial',
                name='output_net')
        else:
            assert self.output_net.dim_in == self.dim_h
        self.output_net.name = self.name + '_output_net'

        if self.conditional is not None:
            assert self.conditional.dim_out == self.dim_h
            self.conditional.name = self.name + '_conditional'

        self.nets = [self.input_net, self.output_net, self.conditional]

    def step_sample_preact(self, h_, c_, x_, *params):
        Ur            = self.get_recurrent_args(*params)[0]
        input_params  = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y      = self.input_net.step_preact(x_, *input_params)
        h, c   = self._step(y, h_, c_, Ur)
        preact = self.output_net.step_preact(h, *output_params)
        return h, c, preact

    def step_sample(self, h_, c_, x_, *params):
        h, c, preact = self.step_sample_preact(h_, c_, x_, *params)

        p    = self.output_net.distribution(preact)
        x, _ = self.output_net.sample(p, n_samples=1)
        x    = x[0]
        return h, c, x, p

    def call_seqs(self, x, condition_on, *params):
        i_params = self.get_input_args(*params)
        a = self.input_net.step_preact(x, *i_params)
        if condition_on is not None:
            a += condition_on
        seqs = [a]
        return seqs

    def _step(self, y, h_, c_, Ur):
        preact = T.dot(h_, Ur) + y
        i, f, o, c = self.get_gates(preact)

        c = f * c_ + i * c
        h = o * T.tanh(c)
        return h, c

    def step_call(self, x, h0, c0, condition_on, *params):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        seqs         = self.call_seqs(x, condition_on, *params)
        outputs_info = [h0, c0]
        non_seqs     = self.get_recurrent_args(*params)

        (h, c), updates = theano.scan(
            self._step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=self.name + '_recurrent_steps',
            n_steps=n_steps,
            strict=True)

        o_params    = self.get_output_args(*params)
        out_net_out = self.output_net.step_call(h, *o_params)
        preact      = out_net_out['z']
        p           = out_net_out['p']
        #y           = self.output_net.sample(p=p)

        return OrderedDict(h=h, p=p, z=preact), updates

    def sample(self, x0=None, h0=None, c0=None, n_samples=10, n_steps=10,
               condition_on=None, debug=False):
        if x0 is None:
            x0, _ = self.output_net.sample(
                p=T.constant(0.5).astype(floatX),
                size=(n_samples, self.output_net.dim_out)).astype(floatX)

        if h0 is None:
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX)
        if c0 is None:
            c0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX)
        z0 = self.output_net.preact(h0)

        seqs = []
        outputs_info = [h0, c0, x0, None]
        non_seqs = []
        step = self.step_sample
        p0 = self.output_net.distribution(z0)

        non_seqs += self.get_sample_params()
        if debug:
            return self.step_sample(h0, x0, *self.get_sample_params())

        outs = scan(step, seqs, outputs_info, non_seqs, n_steps,
                    name=self.name+'_sampling', strict=False)
        (h, c, x, p), updates = outs

        x = concatenate([x0[None, :, :], x])
        h = concatenate([h0[None, :, :], h])
        p = concatenate([p0[None, :, :], p])

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0), updates

    def __call__(self, x, h0=None, c0=None, condition_on=None):
        if h0 is None:
            h0 = T.alloc(0., x.shape[1], self.dim_h).astype(floatX)

        if c0 is None:
            c0 = T.alloc(0., x.shape[1], self.dim_h).astype(floatX)

        params = self.get_sample_params()

        return self.step_call(x, h0, c0, condition_on, *params)