'''
Module for RNN layers
'''

import copy
from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Layer
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
    def __init__(self, dim_in, dim_hs, dim_out=None,
                 conditional=None, input_net=None, output_net=None,
                 name='rnn', **kwargs):

        self.dim_in = dim_in
        self.dim_hs = dim_hs
        self.n_layers = len(self.dim_hs)

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
    def factory(dim_in=None, dim_hs=None, **kwargs):
        return RNN(dim_in, dim_hs, **kwargs)

    @staticmethod
    def mlp_factory(dim_in, dim_hs, data_iter, dim_out=None,
                    i_net=None, a_net=None, o_net=None, c_net=None):
        mlps = {}

        if dim_out is None:
            dim_out = dim_in

        if i_net is not None:
            i_net['distribution'] = 'centered_binomial'
            input_net = MLP.factory(dim_in=dim_in, dim_out=dim_hs[0],
                                    name='input_net', **i_net)
            mlps['input_net'] = input_net

        if o_net is not None:
            o_net['distribution'] = data_iter.distributions[data_iter.name]
            output_net = MLP.factory(dim_in=dim_hs[-1], dim_out=dim_out,
                                     name='output_net', **o_net)
            mlps['output_net'] = output_net

        if c_net is not None:
            if not c_net.get('dim_in', False):
                c_net['dim_in'] = dim_in
            conditional = MLP.factory(dim_out=dim_hs[0],
                                      name='conditional', **c_net)
            mlps['conditional'] = conditional

        return mlps

    def set_params(self):
        self.params = OrderedDict()
        for i, dim_h in enumerate(self.dim_hs):
            Ur = ortho_weight(dim_h)
            self.params['Ur%d' % i] = Ur

        self.set_net_params()

    def set_net_params(self):
        if self.input_net is None:
            self.input_net = MLP(
                self.dim_in, self.dim_hs[0],
                rng=self.rng, trng=self.trng,
                distribution='centered_binomial',
                name='input_net')
        else:
            assert self.input_net.dim_in == self.dim_in
            assert self.input_net.dim_out == self.dim_hs[0]
        self.input_net.name = self.name + '_input_net'

        if self.output_net is None:
            self.output_net = MLP(
                self.dim_hs[-1], self.dim_out,
                rng=self.rng, trng=self.trng,
                distribution='binomial',
                name='output_net')
        else:
            assert self.output_net.dim_in == self.dim_hs[-1]
        self.output_net.name = self.name + '_output_net'

        if self.conditional is not None:
            assert self.conditional.dim_out == self.dim_hs[0]
            self.conditional.name = self.name + '_conditional'

        self.nets = [self.input_net, self.output_net, self.conditional]

        self.inter_nets = []

        for i in xrange(self.n_layers - 1):
            n = MLP(self.dim_hs[i], self.dim_hs[i+1],
                    rng=self.rng, trng=self.trng,
                    distribution='centered_binomial',
                    name='rnn_net%d' % i)

            self.inter_nets.append(n)

    def set_tparams(self):
        tparams = super(RNN, self).set_tparams()
        for net in self.inter_nets + self.nets:
            if net is not None:
                tparams.update(**net.set_tparams())

        self.param_idx = [self.n_layers]
        accum = self.param_idx[0]

        for net in self.inter_nets + self.nets:
            if net is not None:
                accum += len(net.get_params())
            self.param_idx.append(accum)

        return tparams

    def get_params(self):
        params = [self.__dict__['Ur%d' % i] for i in range(self.n_layers)]
        for net in self.inter_nets:
            params += net.get_params()

        return params

    def get_net_params(self):
        params = []
        for net in self.nets:
            if net is not None:
                params += net.get_params()
        return params

    def get_sample_params(self):
        params = self.get_params() + self.get_net_params()
        return params

    def get_recurrent_args(self, *args):
        return args[:self.param_idx[0]]

    def get_inter_args(self, level, *args):
        return args[self.param_idx[level]:self.param_idx[level+1]]

    def get_input_args(self, *args):
        return args[self.param_idx[self.n_layers-1]:self.param_idx[self.n_layers]]

    def get_output_args(self, *args):
        return args[self.param_idx[self.n_layers]:self.param_idx[self.n_layers+1]]

    def get_conditional_args(self, *args):
        return args[self.param_idx[self.n_layers+1]:self.param_idx[self.n_layers+2]]

    # Extra functions ---------------------------------------------------------

    def energy(self, X, h0=None):
        outs, updates = self.__call__(X[:-1], h0=h0)
        p = outs['p']
        energy = self.neg_log_prob(X[1:], p).sum(axis=0)
        return energy

    def neg_log_prob(self, x, p):
        return self.output_net.neg_log_prob(x, p)

    def l2_decay(self, rate):
        cost = sum([rate * (self.__dict__['Ur%d' % i] ** 2).sum()
                    for i in range(self.n_layers)])
        cost += sum([rate * (self.__dict__['W%d' % i] ** 2).sum()
                     for i in range(self.n_layers - 1)])

        for net in self.nets:
            if net is None:
                continue
            cost += net.get_L2_weight_cost(rate)

        rval = OrderedDict(
            cost = cost
        )

        return rval

    def step_sample_preact(self, *params):
        params = list(params)
        hs_ = params[:self.n_layers]
        x = params[self.n_layers]
        params = params[self.n_layers+1:]

        Urs           = self.get_recurrent_args(*params)
        input_params  = self.get_input_args(*params)
        output_params = self.get_output_args(*params)
        y             = self.input_net.step_preact(x, *input_params)

        hs = []
        for i in xrange(self.n_layers):
            h_ = hs_[i]
            Ur = Urs[i]
            h = self._step(1, y, h_, Ur)
            hs.append(h)

            if i < self.n_layers - 1:
                inter_params = self.get_inter_args(i, *params)
                y = self.inter_nets[i].step_preact(h, *inter_params)

        preact = self.output_net.step_preact(h, *output_params)
        return tuple(hs) + (preact,)

    def step_sample(self, *params):
        outs      = self.step_sample_preact(*params)
        hs        = outs[:self.n_layers]
        preact    = outs[-1]
        p         = self.output_net.distribution(preact)
        x, _      = self.output_net.sample(p, n_samples=1)
        x         = x[0]
        return tuple(hs) + (x, p)

    def _step(self, m, y, h_, Ur):
        preact = T.dot(h_, Ur) + y
        h      = T.tanh(preact)
        h      = m * h + (1 - m) * h_
        return h

    def call_seqs(self, x, condition_on, level, *params):
        if level == 0:
            i_params = self.get_input_args(*params)
            a        = self.input_net.step_preact(x, *i_params)
        else:
            i_params = self.get_inter_args(level-1, *params)
            a        = self.inter_nets[level-1].step_preact(x, *i_params)

        if condition_on is not None:
            a += condition_on

        return [a]

    def step_call(self, x, m, h0s, *params):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        hs = []
        for i, h0 in enumerate(h0s):
            seqs         = [m[:, :, None]] + self.call_seqs(x, None, i, *params)
            outputs_info = [h0]
            non_seqs     = [self.get_recurrent_args(*params)[i]]

            h, updates = theano.scan(
                self._step,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=self.name + '_recurrent_steps_%d' % i,
                n_steps=n_steps)
            x = h

        o_params    = self.get_output_args(*params)
        out_net_out = self.output_net.step_call(h, *o_params)
        preact      = out_net_out['z']
        p           = out_net_out['p']

        return OrderedDict(hs=hs, p=p, z=preact), updates

    def __call__(self, x, m=None, h0s=None, condition_on=None):
        if h0s is None:
            h0s = [T.alloc(0., x.shape[1], dim_h).astype(floatX) for dim_h in self.dim_hs]

        if m is None:
            m = T.ones((x.shape[0], x.shape[1])).astype(floatX)

        params = self.get_sample_params()

        return self.step_call(x, m, h0s, *params)

    def sample(self, x0=None, h0s=None, n_samples=10, n_steps=10,
               condition_on=None, debug=False):
        if x0 is None:
            x0, _ = self.output_net.sample(
                p=T.constant(0.5).astype(floatX),
                size=(n_samples, self.output_net.dim_out)).astype(floatX)

        if h0s is None:
            h0s = [T.alloc(0., x.shape[1], dim_h).astype(floatX) for dim_h in self.dim_hs]

        seqs = []
        outputs_info = h0s + [x0, None]
        non_seqs = []
        step = self.step_sample

        non_seqs += self.get_sample_params()

        outs, updates = scan(step, seqs, outputs_info, non_seqs, n_steps,
                             name=self.name+'_sampling', strict=False)
        hs = outs[:self.n_layers]
        x  = outs[self.n_layers]
        p  = outs[self.n_layers+1]

        x  = concatenate([x0[None, :, :], x])
        for i in xrange(self.n_layers):
            hs[i]  = concatenate([h0s[i][None, :, :], hs[i]])
        z0 = self.output_net.preact(h0s[-1])
        p0 = self.output_net.distribution(z0)
        p  = concatenate([p0[None, :, :], p])

        return OrderedDict(x=x, p=p, hs=hs, x0=x0, p0=p0, h0s=h0s), updates

    # Some on-hold functions ------------------------------------------------
    '''
    def step_sample_cond(self, h_, x_, c, *params):
        assert self.conditional is not None

        Ur            = self.get_recurrent_args(*params)[0]
        input_params  = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y = self.input_net.step_preact(x_, *input_params)
        y = y + c
        h = self._step(y, h_, Ur)

        preact = self.output_net.step_preact(h, *output_params)

        p = self.output_net.distribution(preact)
        x = self.output_net.sample(p)
        return h, x, p

    def step_sample_cond_x(self, h_, x_, *params):
        assert self.conditional is not None

        h, preact = self.step_sample_preact(h_, x_, *params)
        c_params  = self.get_conditional_args(*params)
        preact    = preact + self.conditional.step_preact(x_, *c_params)

        p = self.output_net.distribution(preact)
        x = self.output_net.sample(p)
        return h, x, p
    '''