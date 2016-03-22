"""
Module for GRU layers
"""

import copy
from collections import OrderedDict
import numpy as np
import random
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from models.mlp import MLP
from rnn import RNN
from utils import floatX
from utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    norm_weight,
    ortho_weight,
    pi,
    _slice
)


def unpack(dim_in=None,
           dim_h=None,
           h_init=None,
           i_net=None, a_net=None, o_net=None, c_net=None,
           mode=None,
           **model_args):

    # HACKS
    dim_in = int(dim_in)
    dim_h = int(dim_h)
    if i_net is not None:
        i_net = i_net[()]
    if a_net is not None:
        a_net = a_net[()]
    if o_net is not None:
        o_net = o_net[()]
    if c_net is not None:
        c_net = c_net[()]

    trng = RandomStreams(random.randint(0, 100000))

    mlps = GRU.mlp_factory(dim_in, dim_h, mode=mode,
                           i_net=i_net, a_net=a_net, o_net=o_net, c_net=c_net)

    model = GRU(dim_in, dim_h, **mlps)
    models = [model]
    for net in model.nets:
        if net is not None:
            models.append(net)

    if h_init == 'average':
        averager = Averager((dim_h))
        models.append(averager)
    elif h_init == 'mlp':
        h_net = MLP(dim_in, dim_h, dim_h, 1, out_act='T.tanh', name='h_net')
        models.append(h_net)

    return models, model_args, None


class GRU(RNN):
    def __init__(self, dim_in, dim_h, input_net_aux=None, name='gru',
                 **kwargs):

        self.input_net_aux = input_net_aux
        super(GRU, self).__init__(dim_in, dim_h, name=name, **kwargs)

    @staticmethod
    def factory(dim_in=None, dim_h=None, **kwargs):
        return GRU(dim_in, dim_h, **kwargs)

    @staticmethod
    def mlp_factory(dim_in, dim_h, data_iter, a_net=None, **kwargs):
        mlps = RNN.mlp_factory(dim_in, dim_h, data_iter, **kwargs)

        if a_net is not None:
            input_net_aux = MLP.factory(dim_in=dim_in, dim_out=2*dim_h,
                                        name='input_net_aux', **a_net)
            mlps['input_net_aux'] = input_net_aux

        return mlps

    def set_tparams(self):
        tparams = super(GRU, self).set_tparams()

        self.param_idx = [2]
        accum = self.param_idx[0]
        for net in self.nets:
            if net is not None:
                accum += len(net.get_params())
            self.param_idx.append(accum)

        return tparams

    def get_gates(self, x):
        r = T.nnet.sigmoid(_slice(x, 0, self.dim_h))
        u = T.nnet.sigmoid(_slice(x, 1, self.dim_h))
        return r, u

    def set_params(self):
        Ura = np.concatenate([ortho_weight(self.dim_h),
                              ortho_weight(self.dim_h)], axis=1)
        Urb = ortho_weight(self.dim_h)
        self.params = OrderedDict(Ura=Ura, Urb=Urb)
        self.set_net_params()

    def set_net_params(self):
        super(GRU, self).set_net_params()

        if self.input_net_aux is None:
            self.input_net_aux = MLP(
                self.dim_in, 2 * self.dim_h, 2 * self.dim_h, 1,
                rng=self.rng, trng=self.trng,
                h_act='T.nnet.sigmoid', out_act='T.tanh',
                name='input_net_aux')
        else:
            assert self.input_net_aux.dim_in == self.dim_in
            assert self.input_net_aux.dim_out == 2 * self.dim_h
        self.input_net_aux.name = self.name + '_input_net_aux'

        self.nets.append(self.input_net_aux)

    def get_params(self):
        params = [self.Ura, self.Urb]
        return params

    def get_sample_params(self):
        params = [self.Ura, self.Urb]
        for net in self.nets:
            if net is not None:
                params += net.get_params()
        return params

    def get_aux_args(self, *args):
        return args[self.param_idx[3]:self.param_idx[4]]

    def step_sample_preact(self, h_, x_, *params):
        Ura, Urb = self.get_recurrent_args(*params)
        aux_params = self.get_aux_args(*params)
        input_params = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y_aux = self.input_net_aux.step_preact(x_, *aux_params)
        y_input = self.input_net.step_preact(x_, *input_params)

        h = self._step(y_aux, y_input, h_, Ura, Urb)

        preact = self.output_net.step_preact(h, *output_params)
        return h, preact

    '''
    def step_sample_cond(self, h_, x_, c, *params):
        Ura, Urb = self.get_recurrent_args(*params)
        aux_params = self.get_aux_args(*params)
        input_params = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y_aux = self.input_net_aux.preact(x_, *aux_params)
        y_input = self.input_net.preact(x_, *input_params)
        y_input += c

        h = self._step(y_aux, y_input, h_, Ura, Urb)
        preact = self.output_net.preact(h, *output_params)

        p = eval(self.output_net.out_act)(preact)
        x = self.output_net.sample(p)

        return h, x, p
    '''

    def _step(self, y_a, y_i, h_, Ura, Urb):
        preact = T.dot(h_, Ura) + y_a
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + y_i
        h = T.tanh(preactx)
        h = u * h + (1. - u) * h_
        return h

    def call_seqs(self, x, condition_on, *params):
        i_params = self.get_input_args(*params)
        a_params = self.get_aux_args(*params)
        i = self.input_net.step_preact(x, *i_params)
        a = self.input_net_aux.step_preact(x, *a_params)
        if condition_on is not None:
            i += condition_on
        seqs = [a, i]
        return seqs
