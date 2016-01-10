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
from mlp import MLP
from utils import tools
from utils.tools import (
    concatenate,
    floatX,
    init_rngs,
    init_weights,
    norm_weight,
    ortho_weight,
    pi,
    scan
)


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
    def __init__(self, dim_in, dim_h, dim_out=None,
                 conditional=None, input_net=None, output_net=None,
                 name='gen_rnn', **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        if dim_out is None:
            self.dim_out = self.dim_in
        else:
            self.dim_out = dim_out

        self.input_net = input_net
        self.output_net = output_net
        self.conditional = conditional

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        if len(kwargs) > 0:
            print 'Extra args found: %r' % kwargs
        super(RNN, self).__init__(name=name)

    @staticmethod
    def factory(dim_in=None, dim_h=None, **kwargs):
        return RNN(dim_in, dim_h, **kwargs)

    @staticmethod
    def mlp_factory(dim_in, dim_h, dim_out=None,
                    i_net=None, a_net=None, o_net=None, c_net=None):
        mlps = {}

        if dim_out is None:
            dim_out = dim_in

        if i_net is not None:
            input_net = MLP.factory(dim_in=dim_in, dim_out=dim_h,
                                    name='input_net', **i_net)
            mlps['input_net'] = input_net

        if o_net is not None:
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
        self.input_net.name = self.name + '_input_net'

        if self.output_net is None:
            self.output_net = MLP(
                self.dim_h, self.dim_h, self.dim_out, 1,
                rng=self.rng, trng=self.trng,
                h_act='T.nnet.sigmoid', out_act='T.nnet.sigmoid',
                name='output_net')
        else:
            assert self.output_net.dim_in == self.dim_h
        self.output_net.name = self.name + '_output_net'

        if self.conditional is not None:
            assert self.conditional.dim_out == self.dim_h
            self.conditional.name = self.name + '_conditional'

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

    def get_net_params(self):
        params = []
        for net in self.nets:
            if net is not None:
                params += net.get_params()
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

    # Energy functions ---------------------------------------------------------

    def energy(self, X, h0=None):
        outs, updates = self.__call__(X[:-1], h0=h0)
        p = outs['p']
        energy = self.neg_log_prob(X[1:], p).sum(axis=0)
        return energy

    def neg_log_prob(self, x, p):
        return self.output_net.neg_log_prob(x, p)

    # Sample functions ---------------------------------------------------------

    def step_sample_preact(self, h_, x_, *params):
        Ur = self.get_recurrent_args(*params)[0]
        input_params = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y = self.input_net.preact(x_, *input_params)
        h = self._step(y, h_, Ur)

        preact = self.output_net.preact(h, *o_params)
        return h, preact

    def step_sample(self, h_, x_, *params):
        h, preact = self.step_sample_preact(h_, x_, *params)

        p = eval(self.output_net.out_act)(preact)
        x = self.output_net.sample(p)
        return h, x, p

    def step_sample_cond(self, h_, x_, c, *params):
        assert self.conditional is not None
        Ur = self.get_recurrent_args(*params)[0]
        input_params = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y = self.input_net.preact(x_, *input_params)
        y += c
        h = self._step(y, h_, Ur)

        preact = self.output_net.preact(h, *output_params)

        p = eval(self.output_net.out_act)(preact)
        x = self.output_net.sample(p)
        return h, x, p

    def step_sample_cond_x(self, h_, x_, *params):
        assert self.conditional is not None
        h, preact = self.step_sample_preact(h_, x_, *params)
        c_params = self.get_conditional_args(*params)
        preact += self.conditional.preact(x_, *c_params)

        p = eval(self.output_net.out_act)(preact)
        x = self.output_net.sample(p)
        return h, x, p

    def sample(self, x0=None, h0=None, n_samples=10, n_steps=10,
               condition_on=None):
        if x0 is None:
            x0 = self.output_net.sample(
                p=0.5, size=(n_samples, self.output_net.dim_out)).astype(floatX)

        if h0 is None:
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX)
        z0 = self.output_net(h0, return_preact=True)

        seqs = []
        outputs_info = [h0, x0, None]
        non_seqs = []
        if condition_on is None:
            step = self.step_sample
        else:
            step = self.step_sample_cond
            non_seqs.append(condition_on)
            z0 += condition_on
        p0 = eval(self.output_net.out_act)(z0)

        non_seqs += self.get_sample_params()

        (h, x, p), updates = scan(step, seqs, outputs_info, non_seqs, n_steps,
                                  name=self.name+'_sampling', strict=False)

        x = concatenate([x0[None, :, :], x])
        h = concatenate([h0[None, :, :], h])
        p = concatenate([p0[None, :, :], p])

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0), updates

    # Assignment functions -----------------------------------------------------

    def step_energy(self, x, x_p, h_p, *params):
        h, x_s, p = self.step_sample(h_p, x_p, *params)
        energy = self.neg_log_prob(x, p)
        return energy, h, p

    def step_energy_cond(self, x, x_p, h_p, c, *params):
        h, x_s, p = self.step_sample_cond(h_p, x_p, c, *params)
        energy = self.neg_log_prob(x, p)
        return energy, h, p

    def step_scale(self, scaling, counts, idx, alpha, beta):
        counts = T.set_subtensor(counts[idx, T.arange(counts.shape[1])], 1)
        picked_scaling = scaling[idx, T.arange(scaling.shape[1])]
        scaling = scaling / beta
        scaling = T.set_subtensor(scaling[idx, T.arange(scaling.shape[1])], picked_scaling * alpha)
        scaling = T.clip(scaling, 0.0, 1.0)
        return scaling, counts

    def step_assign(self, idx, h_p, counts, scaling, x, alpha, beta, *params):
        energies, h_n, p = self.step_energy(x, x[idx, T.arange(x.shape[1])], h_p, *params)
        energies -= T.log(scaling)

        idx = T.argmin(energies, axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, p, counts, scaling), theano.scan_module.until(T.all(counts))

    def step_assign_cond(self, idx, h_p, counts, scaling, x, alpha, beta, c, *params):
        energies, h_n, p = self.step_energy_cond(x, x[idx, T.arange(x.shape[1])], h_p, c, *params)
        energies -= T.log(scaling)

        idx = T.argmin(energies, axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, p, counts, scaling), theano.scan_module.until(T.all(counts))

    def step_assign_sample(self, idx, h_p, counts, scaling, x, alpha, beta, *params):
        energies, h_n, p = self.step_energy(x, x[idx, T.arange(x.shape[1])], h_p, *params)
        energies -= T.log(scaling)

        e_max = (-energies).max()
        probs = T.exp(-energies - e_max)
        probs = probs
        probs = probs / probs.sum()
        idx = T.argmax(self.trng.multinomial(pvals=probs).astype('int64'), axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, p, counts, scaling), theano.scan_module.until(T.all(counts))

    def get_first_assign(self, x, p0):
        energy = self.neg_log_prob(x, p0)
        idx = T.argmin(energy, axis=0)
        return idx

    def step_assign_call(self, X, h0, condition_on, alpha, beta, steps, sample,
                         select_first, *params):

        o_params = self.get_output_args(*params)
        p0 = self.output_net(h0)

        counts = T.zeros((X.shape[0], X.shape[1])).astype('int64')
        scaling = T.ones((X.shape[0], X.shape[1])).astype('float32')

        if select_first:
            print 'Selecting first best in assignment'
            idx0 = self.get_first_assign(X, p0)
        else:
            print 'Using 0 as first in assignment'
            idx0 = T.zeros((X.shape[1],)).astype('int64')

        counts = T.set_subtensor(counts[idx0, T.arange(counts.shape[1])], 1)
        scaling = T.set_subtensor(scaling[idx0, T.arange(scaling.shape[1])], scaling[0] * alpha)

        seqs = []
        outputs_info = [idx0, h0, None, counts, scaling]
        non_seqs = [X, alpha, beta]

        if condition_on is None:
            step = self.step_assign
        else:
            step = self.step_assign_cond
            non_seqs.append(condition_on)

        non_seqs += params

        (chain, h_chain, p_chain, counts, scalings), updates = scan(
            step, seqs, outputs_info, non_seqs, steps, name='make_chain',
            strict=False)

        chain = concatenate([idx0[None, :], chain], axis=0)
        p_chain = concatenate([p0[None, :, :], p_chain], axis=0)

        return OrderedDict(chain=chain, probs=p_chain, counts=counts[-1], scalings=scalings[-1]), updates

    def assign(self, X, h0=None, condition_on=None, alpha=0.0, beta=1.0,
               steps=None, sample=False, select_first=False):
        if h0 is None:
            h0 = self.trng.normal(avg=0, std=1.0, size=(X.shape[1], self.dim_h)).astype(floatX)

        if steps is None:
            steps = X.shape[0] - 1

        params = self.get_sample_params()

        return self.step_assign_call(X, h0, condition_on, alpha, beta, steps,
                                     sample, select_first, *params)

    # Call functions -----------------------------------------------------------

    def _step(self, y, h_, Ur):
        preact = T.dot(h_, Ur) + y
        h = T.tanh(preact)
        return h

    def call_seqs(self, x, condition_on, *params):
        i_params = self.get_input_args(*params)
        a = self.input_net.preact(x, *i_params)
        if condition_on is not None:
            a += condition_on
        seqs = [a]
        return seqs

    def __call__(self, x, h0=None, condition_on=None):
        if h0 is None:
            h0 = T.alloc(0., x.shape[1], self.dim_h).astype(floatX)

        params = self.get_sample_params()

        return self.step_call(x, h0, condition_on, *params)

    def step_call(self, x, h0, condition_on, *params):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        seqs = self.call_seqs(x, condition_on, *params)
        outputs_info = [h0]
        non_seqs = self.get_recurrent_args(*params)

        h, updates = theano.scan(
            self._step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=self.name + '_recurrent_steps',
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        o_params = self.get_output_args(*params)
        preact = self.output_net.preact(h, *o_params)
        p = eval(self.output_net.out_act)(preact)
        y = self.output_net.sample(p=p)

        return OrderedDict(h=h, y=y, p=p, z=preact), updates
