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

from . import Layer
from .mlp import MLP
from .rnn import RNN
from ..utils import floatX
from ..utils.tools import (
    concatenate,
    init_weights,
    norm_weight,
    ortho_weight,
    _slice
)


class GRU(RNN):
    def __init__(self, dim_in, dim_hs, input_net_aux=None, name='gru',
                 **kwargs):
        '''Init function for GRU.'''

        self.input_net_aux = input_net_aux
        super(GRU, self).__init__(dim_in, dim_hs, name=name, **kwargs)

    @staticmethod
    def factory(data_iter=None, dim_in=None, dim_out=None, dim_hs=None,
                i_net=None, a_net=None, o_net=None, c_net=None, **kwargs):
        '''Factory for creating MLPs for GRU and returning instance.

        Convenience to quickly create MLPs from dictionaries, linking all
        relevent dimensions and distributions.

        Args:
            dim_in: int, input dimention.
            dim_hs: list of int, dimensions of reccurent units.
            data_iter: Dataset object, provides dimension and distribution info.
            dim_out: int (optional), output dimension. If not provided, assumed
                to be dim_in.
            i_net: dict, input network args.
            o_net: dict, output network args.
            c_net: dict, conditional network args.

        Returns:
            GRU: GRU object
        '''

        if dim_in is None:
            dim_in = data_iter.dims[data_iter.name]
        if dim_out is None:
            dim_out = dim_in
        if i_net is None: i_net = dict()
        if o_net is None: o_net = dict()
        if a_net is None: a_net = dict()

        mlps = {}

        i_net['distribution'] = 'centered_binomial'
        input_net = MLP.factory(dim_in=dim_in, dim_out=dim_hs[0],
                                name='input_net', **i_net)

        a_net['distribution'] = 'centered_binomial'
        input_net_aux = MLP.factory(dim_in=dim_in, dim_out=2*dim_hs[0],
                                    name='input_net_aux', **a_net)

        if not o_net.get('distribution', False):
            if data_iter is not None:
                o_net['distribution'] = data_iter.distributions[data_iter.name]
            else:
                o_net['distribution'] = 'binomial'
        output_net = MLP.factory(dim_in=dim_hs[-1], dim_out=dim_out,
                                 name='output_net', **o_net)
        mlps.update(input_net=input_net, output_net=output_net,
                    input_net_aux=input_net_aux)

        if c_net is not None:
            if not c_net.get('dim_in', False):
                c_net['dim_in'] = dim_in
            conditional = MLP.factory(dim_out=dim_hs[0],
                                      name='conditional', **c_net)
            mlps['conditional'] = conditional

        kwargs.update(**mlps)

        return GRU(dim_in, dim_hs, dim_out=dim_out, **kwargs)

    def set_tparams(self):
        tparams = super(GRU, self).set_tparams()

        self.param_idx = [2 * self.n_layers]
        accum = self.param_idx[0]

        for net in self.inter_nets + self.nets:
            if net is not None:
                accum += len(net.get_params())
            self.param_idx.append(accum)
        return tparams

    def get_gates(self, x):
        '''Split gates.'''
        r = T.nnet.sigmoid(_slice(x, 0, x.shape[x.ndim-1] // 2))
        u = T.nnet.sigmoid(_slice(x, 1, x.shape[x.ndim-1] // 2))
        return r, u

    def set_params(self):
        '''Initialize GRU parameters.'''
        self.params = OrderedDict()
        for i, dim_h in enumerate(self.dim_hs):
            Ura = np.concatenate([ortho_weight(dim_h),
                                  ortho_weight(dim_h)], axis=1)
            Urb = ortho_weight(dim_h)
            self.params['Ura%d' % i] = Ura
            self.params['Urb%d' % i] = Urb
        self.set_net_params()

    def set_net_params(self):
        '''Returns MLP parameters for scan.'''
        super(GRU, self).set_net_params()

        if self.input_net_aux is None:
            self.input_net_aux = MLP(
                self.dim_in, 2 * self.dim_h, 2 * self.dim_hs[0], 1,
                rng=self.rng, trng=self.trng,
                h_act='T.nnet.sigmoid', out_act='T.tanh',
                name='input_net_aux')
        else:
            assert self.input_net_aux.dim_in == self.dim_in
            assert self.input_net_aux.dim_out == 2 * self.dim_hs[0]
        self.input_net_aux.name = self.name + '_input_net_aux'

        self.nets.append(self.input_net_aux)

        for i in xrange(self.n_layers - 1):
            n = MLP(self.dim_hs[i], 2 * self.dim_hs[i+1],
                    rng=self.rng, trng=self.trng,
                    distribution='centered_binomial',
                    name='rnn_net_aux%d' % i)
            self.inter_nets.append(n) #insert(2 * i + 1, n)

    def get_params(self):
        '''Returns parameters for scan.'''
        params = []
        for i in range(self.n_layers):
            params += [self.__dict__['Ura%d' % i], self.__dict__['Urb%d' % i]]
        for net in self.inter_nets:
            params += net.get_params()
        return params

    def get_inter_aux_args(self, level, *args):
        return args[self.param_idx[self.n_layers + level]:self.param_idx[self.n_layers + level + 1]]

    def get_input_args(self, *args):
        return args[self.param_idx[2 * (self.n_layers - 1)]:self.param_idx[2 * (self.n_layers - 1) + 1]]

    def get_output_args(self, *args):
        return args[self.param_idx[2 * (self.n_layers - 1) + 1]:self.param_idx[2 * (self.n_layers - 1) + 2]]

    def get_conditional_args(self, *args):
        return args[self.param_idx[2 * (self.n_layers - 1) + 2]:self.param_idx[2 * (self.n_layers - 1) + 3]]

    def get_aux_args(self, *args):
        return args[self.param_idx[2 * (self.n_layers - 1) + 3]:self.param_idx[2 * (self.n_layers - 1) + 4]]

    def step_sample_preact(self, *params):
        '''Returns preact for sampling step.'''

        params = list(params)
        hs_ = params[:self.n_layers]
        x = params[self.n_layers]
        params = params[self.n_layers+1:]

        Urs           = self.get_recurrent_args(*params)
        input_params  = self.get_input_args(*params)
        output_params = self.get_output_args(*params)
        aux_params = self.get_aux_args(*params)

        y_aux = self.input_net_aux.preact(x_, *aux_params)
        y_input = self.input_net.preact(x_, *input_params)

        hs = []
        for i in xrange(self.n_layers):
            h_ = hs_[i]
            Ura, Urb = Urs[2 * i: 2 * i + 2]
            h = self._step(1, y_aux, y_input, h_, Ura, Urb)
            hs.append(h)

            if i < self.n_layers - 1:
                inter_params = self.get_inter_args(i, *params)
                inter_params_aux = self.get_inter_aux_args(i, *params)
                y_aux = self.inter_nets[i].step_preact(h, *inter_params)
                y_input = self.inter_nets[self.n_layers - 1 + i]

        preact = self.output_net.preact(h, *output_params)
        return h, preact

    def _step(self, m, y_a, y_i, h_, Ura, Urb):
        preact = T.dot(h_, Ura) + y_a
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + y_i
        h = T.tanh(preactx)
        h = u * h + (1. - u) * h_
        h = m * h + (1 - m) * h_
        return h

    def step_call(self, x, m, h0s, *params):
        '''Step version of __call__ for scan'''
        n_steps = x.shape[0]
        n_samples = x.shape[1]
        updates = theano.OrderedUpdates()

        hs = []
        for i, h0 in enumerate(h0s):
            seqs         = [m[:, :, None]] + self.call_seqs(x, None, i, *params)
            outputs_info = [h0]
            non_seqs     = self.get_recurrent_args(*params)[2*i:2*i+2]

            h, updates_ = theano.scan(
                self._step,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=self.name + '_recurrent_steps_%d' % i,
                n_steps=n_steps)
            x = h
            hs.append(h)
            updates += updates_

        o_params    = self.get_output_args(*params)
        out_net_out = self.output_net.step_call(h, *o_params)
        preact      = out_net_out['z']
        p           = out_net_out['p']

        return OrderedDict(hs=hs, p=p, z=preact), updates

    def call_seqs(self, x, condition_on, level, *params):
        if level == 0:
            i_params = self.get_input_args(*params)
            a_params = self.get_aux_args(*params)
            i = self.input_net.step_preact(x, *i_params)
            a = self.input_net_aux.step_preact(x, *a_params)
        else:
            i_params = self.get_inter_args(level - 1, *params)
            a_params = self.get_inter_aux_args(level - 1, *params)
            i = self.inter_nets[level - 1].step_preact(x, *i_params)
            a = self.inter_nets[self.n_layers + level - 1].step_preact(x, *i_params)

        if condition_on is not None:
            i += condition_on
        seqs = [a, i]
        return seqs
