'''
Module for RNN layers.
'''

import copy
from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Layer
from .mlp import MLP
from ..utils import floatX, pi, tools
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    ortho_weight,
    norm_weight,
    scan
)


def init_h(h_init, X, batch_size, models, **h_args):
    '''Initializes the RNN hidden state.

    Args:
        h_init: str. Type of initialization.
        X: 3D T.tensor. Input tensor for initialization through MLP.
        batch_size: int
        models: list of Layer, pulls 'h_net' for initialization (TODO change this).
        **h_args: kwargs for different initializations.
    Returns:
        h0s: 3D Tensor of initializations. Full 3D tensor returned to train `h_net`.
    '''
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
    '''RNN class.

    Implements a generic multilayer RNN.

    Attributes:
        dim_in: int, input dimension.
        dim_out: int, output dimension.
        dim_hs: list of int: dimenstions of recurrent units.
        n_layers: int, number of recurrent layers. Should match len(dim_hs).
        input_net: MLP object, MLP to feed input into recurrent layers.
        output_net: MLP object, MLP to read from recurrent layers.
        condtional: MLP object (optional), MLP to condition output on previous
            output.
    '''

    def __init__(self, dim_in, dim_hs, dim_out=None,
                 conditional=None, input_net=None, output_net=None,
                 name='rnn', **kwargs):
        '''Init function for RNN.

        Args:
            dim_in: int, input dimension.
            dim_hs: list of int, dimensions of the recurrent layers.
            dim_out: int, output dimention.
            conditional: MLP (optional), conditional network for p(x_t | x_{t-1})
            input_net: MLP, input network.
            output_net: MLP, output network.
        '''

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
    def factory(data_iter=None, dim_in=None, dim_out=None, dim_hs=None,
                i_net=None, o_net=None, c_net=None, **kwargs):
        '''Factory for creating MLPs for RNN and returning .

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
            RNN: RNN object

        '''

        if dim_in is None:
            dim_in = data_iter.dims[data_iter.name]
        if dim_out is None:
            dim_out = dim_in
        if i_net is None: i_net = dict()
        if o_net is None: o_net = dict()

        mlps = {}
        i_net['distribution'] = 'centered_binomial'
        input_net = MLP.factory(dim_in=dim_in, dim_out=dim_hs[0],
                                name='input_net', **i_net)

        if not o_net.get('distribution', False):
            o_net['distribution'] = data_iter.distributions[data_iter.name]
        output_net = MLP.factory(dim_in=dim_hs[-1], dim_out=dim_out,
                                 name='output_net', **o_net)
        mlps.update(input_net=input_net, output_net=output_net)

        if c_net is not None:
            if not c_net.get('dim_in', False):
                c_net['dim_in'] = dim_in
            conditional = MLP.factory(dim_out=dim_hs[0],
                                      name='conditional', **c_net)
            mlps['conditional'] = conditional

        kwargs.update(**mlps)

        return RNN(dim_in, dim_hs, dim_out=dim_out, **kwargs)

    def set_params(self):
        '''Initialize RNN parameters.'''
        self.params = OrderedDict()
        for i, dim_h in enumerate(self.dim_hs):
            Ur = ortho_weight(dim_h)
            self.params['Ur%d' % i] = Ur

        self.set_net_params()

    def set_net_params(self):
        '''Initialize MLP parameters.'''
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
        '''Sets and returns theano parameters.'''
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
        '''Returns parameters for scan.'''
        params = [self.__dict__['Ur%d' % i] for i in range(self.n_layers)]
        for net in self.inter_nets:
            params += net.get_params()

        return params

    def get_net_params(self):
        '''Returns MLP parameters for scan.'''
        params = []
        for net in self.nets:
            if net is not None:
                params += net.get_params()
        return params

    def get_sample_params(self):
        '''Returns parameters used for sampling.'''
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

    def energy(self, X, h0s=None):
        '''Negative log probability of data point.'''
        outs, updates = self.__call__(X[:-1], h0s=h0s)
        p = outs['p']
        energy = self.neg_log_prob(X[1:], p).sum(axis=0)
        return energy

    def neg_log_prob(self, x, p):
        '''Negative log prob function.'''
        return self.output_net.neg_log_prob(x, p)

    def l2_decay(self, rate):
        '''L2 decay.'''
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
        '''Returns preact for sampling step.'''
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
        '''RNN step sample function for scan.

        A convenience function for scan, this method samples the output given
        the input, returning the current states and sample.

        '''
        outs   = self.step_sample_preact(*params)
        hs     = outs[:self.n_layers]
        preact = outs[-1]
        p      = self.output_net.distribution(preact)
        x, _   = self.output_net.sample(p, n_samples=1)
        x      = x[0]
        return tuple(hs) + (x, p, preact)

    def _step(self, m, y, h_, Ur):
        preact = T.dot(h_, Ur) + y
        h      = T.tanh(preact)
        h      = m * h + (1 - m) * h_
        return h

    def call_seqs(self, x, condition_on, level, *params):
        '''Prepares the input for __call__'''
        if level == 0:
            i_params = self.get_input_args(*params)
            a        = self.input_net.step_preact(x, *i_params)
        else:
            i_params = self.get_inter_args(level - 1, *params)
            a        = self.inter_nets[level - 1].step_preact(x, *i_params)

        if condition_on is not None:
            a += condition_on

        return [a]

    def step_call(self, x, m, h0s, *params):
        '''Step version of __call__ for scan'''
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        updates = theano.OrderedUpdates()

        hs = []
        for i, h0 in enumerate(h0s):
            seqs         = [m[:, :, None]] + self.call_seqs(x, None, i, *params)
            outputs_info = [h0]
            non_seqs     = [self.get_recurrent_args(*params)[i]]
            h, updates_ = theano.scan(
                self._step,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=self.name + '_recurrent_steps_%d' % i,
                n_steps=n_steps)
            hs.append(h)
            x = h
            updates += updates_

        o_params    = self.get_output_args(*params)
        out_net_out = self.output_net.step_call(h, *o_params)
        preact      = out_net_out['z']
        p           = out_net_out['p']

        return OrderedDict(hs=hs, p=p, z=preact), updates

    def __call__(self, x, m=None, h0s=None, condition_on=None):
        '''Call function.

        For learning RNNs.

        Args:
            x: 3D T.tensor, input sequence. window x batch x dim
            m: T.tensor, mask. window x batch. For masking in recurrent steps.
            h0s: list of T.tensor (optional), initial h0s.
            condition_on: T.tensor (optional), conditional for recurrent step.
        Returns:
            results: OrderedDict of hiddens, probabilities, and preacts.
            updates: OrderedUpdates.
        '''
        if h0s is None:
            h0s = [T.alloc(0., x.shape[1], dim_h).astype(floatX) for dim_h in self.dim_hs]

        if m is None:
            m = T.ones((x.shape[0], x.shape[1])).astype(floatX)

        params = self.get_sample_params()

        return self.step_call(x, m, h0s, *params)

    def sample(self, x0=None, h0s=None, n_samples=10, n_steps=10,
               condition_on=None, debug=False):
        '''Samples from an initial state.

        Args:
            x0: T.tensor (optional), initial input state.
            h0: T.tensor (optional), initial recurrent state.
            n_samples: int (optional), if no x0 or h0, used to initial batch.
                Number of chains.
            n_steps: int, number of sampling steps.
        Returns:
            results: OrderedDict of samples, probs, recurrent states, etc.
            updates: OrderedUpdates.
        '''
        if x0 is None:
            x0, _ = self.output_net.sample(
                p=T.constant(0.5).astype(floatX),
                size=(n_samples, self.output_net.dim_out)).astype(floatX)

        if h0s is None:
            h0s = [T.alloc(0., x.shape[1], dim_h).astype(floatX) for dim_h in self.dim_hs]

        seqs = []
        outputs_info = h0s + [x0, None, None]
        non_seqs = []
        non_seqs += self.get_sample_params()

        if n_steps == 1:
            inps = outputs_info[:-2] + non_seqs
            outs = self.step_sample(*inps)
            updates = theano.OrderedUpdates()
            hs = outs[:self.n_layers]
            x, p, z = outs[-3:]
            x = T.shape_padleft(x)
            p = T.shape_padleft(p)
            z = T.shape_padleft(z)
            hs = [T.shape_padleft(h) for h in hs]
        else:
            outs, updates = scan(self.step_sample, seqs, outputs_info, non_seqs,
                                 n_steps, name=self.name+'_sampling', strict=False)
            hs = outs[:self.n_layers]
            x, p, z = outs[-3:]

        return OrderedDict(x=x, p=p, z=z, hs=hs), updates


class SimpleRNN(RNN):
    '''Simple RNN class, single hidden layer.

    Wraps RNN but with a single hidden layer in __init__ instead of list.

    '''
    def __init__(self, dim_in, dim_h, **kwargs):
        '''SimpleRNN init function.'''
        super(SimpleRNN, self).__init__(dim_in, [dim_h], **kwargs)

    @staticmethod
    def factory(data_iter=None, dim_in=None, dim_out=None, dim_h=None,
                    i_net=None, o_net=None, c_net=None, **kwargs):
        '''Convenience factory for SimpleRNN (see `RNN.factory`).'''

        if dim_in is None:
            dim_in = data_iter.dims[data_iter.name]
        if dim_out is None:
            dim_out = dim_in
        if i_net is None: i_net = dict()
        if o_net is None: o_net = dict()

        mlps = {}

        i_net['distribution'] = 'centered_binomial'
        input_net = MLP.factory(dim_in=dim_in, dim_out=dim_h,
                                name='input_net', **i_net)

        o_net['distribution'] = data_iter.distributions[data_iter.name]
        output_net = MLP.factory(dim_in=dim_h, dim_out=dim_out,
                                 name='output_net', **o_net)
        mlps.update(input_net=input_net, output_net=output_net)

        if c_net is not None:
            if not c_net.get('dim_in', False):
                c_net['dim_in'] = dim_in
            conditional = MLP.factory(dim_out=dim_h,
                                      name='conditional', **c_net)
            mlps['conditional'] = conditional

        kwargs.update(**mlps)

        return SimpleRNN(dim_in, dim_h, dim_out=dim_out, **kwargs)

    def energy(self, X, h0=None):
        '''Energy function.'''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).energy(X, h0s=h0s)

    def __call__(self, x, m=None, h0=None, condition_on=None):
        '''Call function (see `RNN.__call__`).'''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).__call__(
            x, m=m, h0s=h0s, condition_on=condition_on)

    def sample(self, x0=None, h0=None, **kwargs):
        '''Sample the SimpleRNN (see `RNN.sample`).'''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).sample(x0=x0, h0s=h0s, **kwargs)
