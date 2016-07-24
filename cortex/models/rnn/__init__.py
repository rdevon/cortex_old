'''
Module for RNN layers.
'''

import copy
from collections import OrderedDict
import numpy as np
import pprint
import theano
from theano import tensor as T

from .. import Cell, norm_weight, ortho_weight
from ..extra_layers import Averager
from .. import mlp as mlp_module
from ...costs import squared_error
from ...utils import concatenate, floatX, pi, scan
from ...utils.tools import _p

def unpack(rnn_args, **model_args):
    '''Unpacks a saved RNN.

    See `utils.tools.load_model` for details.

    Args:
        rnn_args (dict): dictionary of model arguments for forming object.
        **model_args: keyword arguments of saved parameters.

    Returns:
        list: list of models.
        dict: dictionary of saved parameters.

    '''

    model = factory(**rnn_args)
    models = [model] + model.get_components()
    return models, model_args, None


class RNNInitializer(Cell):
    '''Initializer for RNNs.

    Currently supports MLP intialization and averager.

    Attributes:
        initialization (str): intialization type.
        dim_in (int): input dimension. For MLP.
        dim_outs (list): hidden dimensions.

    '''
    _required = ['dim_in', 'dim_out']
    _dim_map = {
        'input': 'dim_in',
        'output': 'dim_out'
    }

    def __init__(self, dim_in, dim_out, initialization='MLP',
                 name='rnn_initializer', **kwargs):
        '''Initialization function for RNN_Initializer.

        Args:
            dim_in (int): input dimension. For MLP.
            dim_outs (list): hidden dimensions.
            initialization (str): intialization type.
            **kwargs: keyword arguments for initialization.

        '''
        self.initialization = initialization
        self.dim_in = dim_in
        self.dim_out = dim_out

        super(RNNInitializer, self).__init__(name=name)

    def set_components(self, **kwargs):
        if self.initialization is None:
            init_args = None
        elif self.initialization == 'MLP':
            init_args = {
                'cell_type': 'MLP',
                'dim_in': self.dim_in,
                'dim_out': self.dim_out,
                '_passed': ['dim_h', 'dim_hs', 'n_layers', 'h_act'],
                '_required': {'out_act': 'tanh'}
            }
        elif self.initialization == 'Averager':
            init_args = {'dim_out': self.dim_out}
        else:
            raise TypeError()

        components = {'initializer': init_args}

        return super(RNNInitializer, self).set_components(
            components=components, **kwargs)

    def _cost(self, X, H):
        '''Call function for RNN_Initializer.

        Updates the initializer or returns cost.

        Args:
            X (T.tensor): input
            hs (list): hidden states

        Returns:
            OrderedDict: cost, mlp output, hs
            theano.OrderedUpdates: updates
            list: constants

        '''
        H = H.copy()
        constants = [H]

        if self.initialization == 'MLP':
            Y = self.initializer(X)['Y']
            cost = squared_error(H, Y)
            return OrderedDict(cost=cost, constants=constants)
        elif self.initialization == 'Averager':
            updates = self.initializer(H)
            return OrderedDict(updates=updates, constants=constants)

    def _feed(self, X, *params):
        '''Initialize the hidden states.

        Args:
            X (T.tensor): input.

        Returns:
            list: initial states.

        '''
        if self.initialization == 'MLP':
            outs = self.initializer._feed(X, *params)
            outs['output'] = outs['Y']
        elif self.initialization == 'Averager':
            outs = {'output': params[0]}
        return outs


class RecurrentUnit(Cell):
    _required = ['dim_h']
    _options = {'weight_noise': False}
    _args = ['dim_h']
    _weights = ['W']
    _dim_map = {
        'input': 'dim_h',
        'output': 'dim_h'
    }

    def __init__(self, dim_h, name='RU', **kwargs):
        self.dim_h = dim_h
        super(RecurrentUnit, self).__init__(name=name, **kwargs)

    def init_params(self):
        '''Initialize RNN parameters.

        '''
        self.params = OrderedDict()
        W = ortho_weight(self.dim_h)
        self.params['W'] = W

    def _recurrence(self, m, y, h_, W):
        '''Recurrence function.

        Args:
            m (T.tensor): masks.
            y (T.tensor): inputs.
            h_ (T.tensor): recurrent state.
            W (theano.shared): recurrent weights.

        Returns:
            T.tensor: next recurrent state.

        '''
        preact = T.dot(h_, W) + y
        h      = T.tanh(preact)
        h      = m * h + (1 - m) * h_
        return h

    def _feed(self, X, M, H0, *params):
        n_steps = X.shape[0]
        seqs         = [M[:, :, None], X]
        outputs_info = [H0]
        non_seqs     = params

        h, updates = theano.scan(
            self._recurrence,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=self.name + '_recurrent_steps',
            n_steps=n_steps)

        return OrderedDict(H=h, updates=updates)


class RNN(Cell):
    '''RNN class.

    Implements a generic multilayer RNN.

    Attributes:
        dim_in (int): input dimension.
        dim_out (int): output dimension.
        dim_hs (list): dimenstions of recurrent units.
        n_layers (int): number of recurrent layers. Should match len(dim_hs).
        input_net (MLP): MLP to feed input into recurrent layers.
        init_net (RNN_Initializer): Initializer for RNN recurrent state.
        nets (list): list of networks. input network, output_net, conditional.
        inter_nets (list): list of inter-networks between recurrent layers.

    '''
    _components = {
        'initializer': {
            'cell_type': 'RNNInitializer',
            '_passed': ['initialization']
        },
        'RU': {
            'cell_type': 'RecurrentUnit',
            '_passed': ['dim_h']
        },
        'input_net': {
            'cell_type': 'MLP',
            '_required': {'out_act': 'tanh'},
            '_passed': ['dim_in']
        }
    }
    _links = [
        ('input_net.output', 'RU.input'),
        ('initializer.output', 'RU.input'),
        ('initializer.input', 'input_net.input')]
    _dim_map = {
        'input': 'dim_in',
        'output': 'dim_h'
    }
    _test_order = ['Y', 'H0', 'H', 'output']

    def __init__(self, name='RNN', **kwargs):
        '''Init function for RNN.

        Args:
            **kwargs: additional keyword arguments.

        '''
        super(RNN, self).__init__(name=name, **kwargs)

    def init_args(self, X, M=None):
        if M is None:
            M = T.ones((X.shape[0], X.shape[1])).astype(floatX)
        return (X, M)

    def _feed(self, X, M, *params):
        ru_params = self.select_params('RU', *params)
        input_params = self.select_params('input_net', *params)
        initializer_params = self.select_params('initializer', *params)

        outs = self.input_net._feed(X, *input_params)
        outs_init = self.initializer._feed(X[0], *initializer_params)
        outs.update(**dict((_p('initializer', k), v)
            for k, v in outs_init.iteritems()))
        outs['H0'] = outs_init['output']
        outs.update(**self.RU(outs['Y'], M, outs_init['output']))
        outs['output'] = outs['H'][-1]
        return outs

class GenRNN(RNN):
    _components = RNN._components
    _components.update(**
        {
            'output_net': {
                'cell_type': 'DistributionMLP',
                '_passed': ['distribution_type']
            }
        })
    _links = RNN._links
    _links += [('output_net.output', )]
    # Step functions -----------------------------------------------------------
    def step_sample_preact(self, *params):
        '''Returns preact for sampling step.

        '''
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

    def __call__(self, X, outputs, diff=False):
        '''Default cost for RNN.

        Negative log likelihood.

        '''
        if diff:
            self.logger.debug('Calculating RNN cost with difference')
            dmu, scale = self.output_net.distribution.split_prob(p)
            mu = X[:-1] + dmu
            p = rnn.output_net.distribution.get_prob(mu, scale)

        p = outputs['p'][:-1]

        cost = self.neg_log_prob(X[1:], p).sum(axis=0).mean()
        return cost

    def _feed(self, x, m, h0s, *params):
        '''Step version of feed for scan

        Args:
            x (T.tensor): input.
            m (T.tensor): mask.
            h0s (list): list of recurrent initial states.
            *params: list of theano.shared.

        Returns:
            OrderedDict: dictionary of results.

        '''
        n_samples = x.shape[1]

        updates = theano.OrderedUpdates()

        x_in = x
        hs = []
        for i, h0 in enumerate(h0s):

            hs.append(h)
            x = h
            updates += updates_

        o_params    = self.get_output_args(*params)
        out_net_out = self.output_net.step_call(h, *o_params)
        preact      = out_net_out['z']
        p           = out_net_out['p']
        error       = self.neg_log_prob(x_in[1:], p[:-1], sum_probs=False)

        return OrderedDict(hs=hs, p=p, error=error, z=preact), updates

    def sample(self, x0=None, h0s=None, n_samples=10, n_steps=10):
        '''Samples from an initial state.

        Args:
            x0 (Optional[T.tensor]): initial input state.
            h0 (Optional[T.tensor]): initial recurrent state.
            n_samples (Optional[int]): if no x0 or h0, used to initial batch.
                Number of chains.
            n_steps (int): number of sampling steps.

        Returns:
            OrderedDict: dictionary of results. hiddens, probabilities, and preacts.
            theano.OrderedUpdates: updates.

        '''
        if x0 is None:
            x0, _ = self.output_net.sample(
                T.constant(0.5).astype(floatX), n_samples=n_samples)

        if h0s is None and self.init_net is not None:
            h0s = self.init_net.initialize(x0)
        elif h0s is None:
            h0s = [T.alloc(0., x0.shape[0], dim_h).astype(floatX)
                   for dim_h in self.dim_hs]

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
    def mlp_factory(dim_in, dim_out, dim_h, **kwargs):
        return RNN.mlp_factory(dim_in, dim_out, [dim_h], **kwargs)

    def energy(self, X, h0=None):
        '''Energy function.

        '''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).energy(X, h0s=h0s)

    def feed(self, x, m=None, h0=None, condition_on=None):
        '''Feed function (see `RNN.feed`).

        '''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).feed(
            x, m=m, h0s=h0s, condition_on=condition_on)

    def sample(self, x0=None, h0=None, **kwargs):
        '''Sample the SimpleRNN (see `RNN.sample`).

        '''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).sample(x0=x0, h0s=h0s, **kwargs)

_classes = {'RNNInitializer': RNNInitializer, 'RecurrentUnit': RecurrentUnit, 'RNN': RNN}