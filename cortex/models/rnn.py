'''
Module for RNN layers.
'''

import copy
from collections import OrderedDict
import numpy as np
import pprint
import theano
from theano import tensor as T

from . import Layer
from .layers import Averager
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


def resolve(c, simple=True):
    from .gru import GRU, SimpleGRU
    if simple:
        c_dict = dict(
            rnn=SimpleRNN,
            gru=SimpleGRU
        )
    else:
        c_dict = dict(
            rnn=RNN,
            gru=GRU
        )
    C = c_dict.get(c, None)

    if C is None:
        raise ValueError('No RNN class `%s`' % c)
    return C

def factory(rnn_type=None, dim_h=None, dim_hs=None, **kwargs):
    if dim_h is None:
        simple = False
        assert dim_hs is not None
    else:
        simple = True
        assert dim_h is not None

    if isinstance(rnn_type, type) and (rnn.__name__ == 'RNN'
                                       or rnn.__base__ == RNN):
        C = rnn_type
    else:
        C = resolve(rnn_type, simple=simple)

    if simple:
        return C.factory(dim_h=dim_h, **kwargs)
    else:
        return C.factory(dim_hs=dim_hs, **kwargs)

def unpack(rnn_args, data_iter=None, **model_args):
    '''Unpacks a saved RNN.

    See `utils.tools.load_model` for details.

    Args:
        rnn_args (dict): dictionary of model arguments for forming object.
        **model_args: keyword arguments of saved parameters.

    Returns:
        list: list of models.
        dict: dictionary of saved parameters.

    '''
    if data_iter is None:
        raise ValueError('Data iterator must be passed to `unpack` '
                         '(or `load_model`).')

    model = RNN.factory(data_iter=data_iter, **rnn_args)
    models = [model] + model.nets + model.inter_nets
    return models, model_args, None


class RNN_initializer(Layer):
    '''Initializer for RNNs.

    Currently supports MLP intialization and averager.

    Attributes:
        initialization (str): intialization type.
        dim_in (int): input dimension. For MLP.
        dim_outs (list): hidden dimensions.
        layers (list): layers for initialization of RNN layers.

    '''
    _components = ['layers']

    def __init__(self, dim_in, dim_outs, initialization='mlp', **kwargs):
        '''Initialization function for RNN_Initializer.

        Args:
            dim_in (int): input dimension. For MLP.
            dim_outs (list): hidden dimensions.
            initialization (str): intialization type.
            **kwargs: keyword arguments for initialization.

        '''
        self.initialization = initialization
        self.dim_in = dim_in
        self.dim_outs = dim_outs

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(RNN_initializer, self).__init__(name='rnn_initializer')

        self.layers = []
        for i, dim_out in enumerate(self.dim_outs):
            if initialization == 'mlp':
                layer = MLP(
                    self.dim_in, dim_out,
                    rng=self.rng, trng=self.trng,
                    distribution='centered_binomial',
                    name='rnn_initializer_mlp_%d' % i,
                    **kwargs)
            elif initialization == 'average':
                layer = Averager((dim_out,), name='rnn_initializer_averager',
                    **kwargs)
            else:
                raise ValueError()
            self.layers.append(layer)

    def set_tparams(self):
        tparams = super(RNN_initializer, self).set_tparams()

        for layer in self.layers:
            if self.initialization in ['mlp', 'average']:
                tparams.update(**layer.set_tparams())

        return tparams

    def get_decay_params(self):
        decay_params = OrderedDict()
        if self.initialization == 'mlp':
            for layer in self.layers:
                decay_params.update(**layer.get_decay_params())
        return decay_params

    def __call__(self, X, hs):
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
        hs = [h.copy() for h in hs]
        cost = T.constant(0.).astype(floatX)
        updates = theano.OrderedUpdates()
        constants = hs

        for i, layer in enumerate(self.layers):
            if self.initialization == 'mlp':
                p = layer.feed(X)
                cost += layer.neg_log_prob(hs[i], p).mean()
            elif self.initialization == 'average':
                updates += layer(hs[i])
            else:
                raise ValueError()

        return OrderedDict(cost=cost, p=p, hs=hs), updates, constants

    def initialize(self, X):
        '''Initialize the hidden states.

        Args:
            X (T.tensor): input.

        Returns:
            list: initial states.

        '''
        if self.initialization == 'mlp':
            return [layer.feed(X) for layer in self.layers]
        elif self.initialization == 'average':
            return [layer.m for layer in self.layers]
        else:
            raise ValueError()


class RNN(Layer):
    '''RNN class.

    Implements a generic multilayer RNN.

    Attributes:
        dim_in (int): input dimension.
        dim_out (int): output dimension.
        dim_hs (list): dimenstions of recurrent units.
        n_layers (int): number of recurrent layers. Should match len(dim_hs).
        input_net (MLP): MLP to feed input into recurrent layers.
        output_net (MLP): MLP to read from recurrent layers.
        condtional (Optional[MLP]): MLP to condition output on previous
            output.
        init_net (RNN_initializer): Initializer for RNN recurrent state.
        nets (list): list of networks. input network, output_net, conditional.
        inter_nets (list): list of inter-networks between recurrent layers.

    '''
    _components = ['nets', 'inter_nets', 'init_net']

    def __init__(self, dim_in, dim_hs, dim_out=None, init_net=None,
                 conditional=None, input_net=None, output_net=None,
                 name='rnn', **kwargs):
        '''Init function for RNN.

        Args:
            dim_in (int): input dimension.
            dim_hs (list): dimensions of the recurrent layers.
            dim_out (int): output dimention.
            conditional (Optional[MLP]): conditional network for p(x_t | x_{t-1})
            input_net (MLP): input network.
            output_net (MLP): output network.

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
        self.init_net = init_net

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(RNN, self).__init__(name=name, **kwargs)

    @staticmethod
    def mlp_factory(dim_in, dim_out, dim_hs, o_dim_in=None, i_net=None,
                    o_net=None, c_net=None, data_distribution='binomial',
                    initialization=None, init_args=None, **kwargs):
        '''Factory for creating MLPs for RNN.

        Args:
            dim_in (int): input dimention.
            dim_out (int): output dimension. If not provided, assumed
                to be dim_in.
            dim_hs (list): dimensions of recurrent units.
            o_dim_in (Optional[int]): optional input dimension for output
                net. If not provided, then use the last hidden dim.
            i_net (dict): input network args.
            o_net (dict): output network args.
            c_net (dict): conditional network args.
            data_distribution (str): distribution of the output.
            initialization (str): type of initialization.
            init_args (dict): initialization keyword arguments.
            **kwargs: extra keyword arguments.

        Returns:
            dict: MLPs.
            dict: extra keyword arguments.

        '''
        import logging
        logger = logging.getLogger('cortex')

        mlps = {}

        # Input network
        if i_net is None: i_net = dict()
        i_net.update(dim_in=dim_in, dim_out=dim_hs[0], name='input_net',
            distribution='centered_binomial')
        logger.debug('Forming RNN with input network parameters %s'
                     % pprint.pformat(i_net))
        input_net = MLP.factory(**i_net)

        # Output network
        if o_dim_in is None:
            o_dim_in = dim_hs[-1]
        if o_net is None: o_net = dict()
        if not o_net.get('distribution', False):
            o_net['distribution'] = data_distribution
        o_net.update(dim_in=o_dim_in, dim_out=dim_out, name='output_net')
        logger.debug('Forming RNN with output network parameters %s'
                     % pprint.pformat(o_net))
        output_net = MLP.factory(**o_net)
        mlps.update(input_net=input_net, output_net=output_net)

        # Conditional network
        if c_net is not None:
            if not c_net.get('dim_in', False):
                c_net['dim_in'] = dim_in
            c_net.update(dim_out=dim_hs[0], name='conditional')
            logger.debug('Forming RNN with conditional network parameters %s'
                % pprint.pformat(c_net))
            conditional = MLP.factory(**c_net)
            mlps['conditional'] = conditional

        # Intitialization
        if initialization is not None:
            if init_args is None: init_args = dict()
            logger.debug('Initializing RNN with %s and parameters %s'
                % (initialization, pprint.pformat(init_args)))
            init_net = RNN_initializer(dim_in, dim_hs,
                                       initialization=initialization,
                                       **init_args)
            mlps['init_net'] = init_net

        return mlps, kwargs

    @staticmethod
    def factory(dim_in=None, dim_out=None, dim_hs=None, **kwargs):
        '''Factory for creating MLPs for RNN and returning .

        Convenience to quickly create MLPs from dictionaries, linking all
        relevent dimensions and distributions.

        Args:
            dim_in (int): input dimention.
            dim_hs (list): dimensions of recurrent units.
            dim_out (Optional[int]): output dimension. If not provided, assumed
                to be dim_in.

        Returns:
            RNN

        '''
        assert len(dim_hs) > 0
        if dim_out is None:
            dim_out = dim_in
        mlps, kwargs = RNN.mlp_factory(dim_in, dim_out, dim_hs, **kwargs)
        kwargs.update(**mlps)

        return RNN(dim_in, dim_hs, dim_out=dim_out, **kwargs)

    def set_params(self):
        '''Initialize RNN parameters.

        '''
        self.params = OrderedDict()
        for i, dim_h in enumerate(self.dim_hs):
            Ur = ortho_weight(dim_h)
            self.params['Ur%d' % i] = Ur

        self.set_net_params()

    def set_net_params(self):
        '''Initialize MLP parameters.

        '''
        assert self.input_net.dim_in == self.dim_in
        assert self.input_net.dim_out == self.dim_hs[0]
        self.input_net.name = self.name + '_input_net'

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
            self.decay_params += n.decay_params

            self.inter_nets.append(n)

    def get_decay_params(self):
        decay_keys = self.params.keys()
        decay_params = OrderedDict((self.name + '.' + k, self.__dict__[k])
            for k in decay_keys)
        for net in self.nets + self.inter_nets:
            if net is not None:
                decay_params.update(**net.get_decay_params())

        decay_params.update(**self.init_net.get_decay_params())
        return decay_params

    def set_tparams(self):
        '''Sets and returns theano parameters.

        '''
        tparams = super(RNN, self).set_tparams()

        if self.init_net is not None:
            tparams.update(**self.init_net.set_tparams())

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
        '''Returns parameters for scan.

        '''
        params = [self.__dict__['Ur%d' % i] for i in range(self.n_layers)]
        if self.weight_noise:
            params = [p + self.trng.normal(
                std=self.weight_nose, size=p.shape, dtype=p.dtype)
                      for p in params]
        for net in self.inter_nets:
            params += net.get_params()

        return params

    def get_net_params(self):
        '''Returns MLP parameters for scan.

        '''
        params = []
        for net in self.nets:
            if net is not None:
                params += net.get_params()
        return params

    def get_sample_params(self):
        '''Returns parameters used for sampling.

        '''
        params = self.get_params() + self.get_net_params()
        return params

    def get_recurrent_args(self, *args):
        '''Get the recurrent arguments for `scan`.

        '''
        return args[:self.param_idx[0]]

    def get_inter_args(self, level, *args):
        '''Get the inter-network arguments for `scan`.

        '''
        return args[self.param_idx[level]:self.param_idx[level+1]]

    def get_input_args(self, *args):
        '''Get the input arguments for `scan`.

        '''
        return args[self.param_idx[self.n_layers-1]
                    :self.param_idx[self.n_layers]]

    def get_output_args(self, *args):
        '''Get the output arguments for `scan`.

        '''
        return args[self.param_idx[self.n_layers]
                    :self.param_idx[self.n_layers+1]]

    def get_conditional_args(self, *args):
        '''Get the conditional arguments for `scan`.

        '''
        return args[self.param_idx[self.n_layers+1]
                    :self.param_idx[self.n_layers+2]]

    # Extra functions ---------------------------------------------------------
    def energy(self, X, h0s=None):
        '''Negative log probability of data point.

        Args:
            X (T.tensor): 3D tensor of samples.
            h0s (list): List of initial hidden states.

        Returns:
            T.tensor: energies for each batch.

        '''
        outs, updates = self.__call__(X[:-1], h0s=h0s)
        p = outs['p']
        energy = self.neg_log_prob(X[1:], p).sum(axis=0)
        return energy

    def neg_log_prob(self, x, p):
        '''Negative log prob function.

        Args:
            x (T.tensor): samples
            p (T.tensor): probabilities

        Returns:
            T.tensor: negative log probabilities.

        '''
        return self.output_net.neg_log_prob(x, p)

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

    def _step(self, m, y, h_, Ur):
        '''Step function for RNN call.

        Args:
            m (T.tensor): masks.
            y (T.tensor): inputs.
            h_ (T.tensor): recurrent state.
            Ur (theano.shared): recurrent connection.

        Returns:
            T.tensor: next recurrent state.

        '''
        preact = T.dot(h_, Ur) + y
        h      = T.tanh(preact)
        h      = m * h + (1 - m) * h_
        return h

    def call_seqs(self, x, condition_on, level, *params):
        '''Prepares the input for `__call__`.

        Args:
            x (T.tensor): input
            condtion_on (T.tensor or None): tensor to condition recurrence on.
            level (int): reccurent level.
            *params: list of theano.shared.

        Returns:
            list: list of scan inputs.

        '''
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
        '''Step version of __call__ for scan

        Args:
            x (T.tensor): input.
            m (T.tensor): mask.
            h0s (list): list of recurrent initial states.
            *params: list of theano.shared.

        Returns:
            OrderedDict: dictionary of results.

        '''
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
            x (T.tensor): input sequence. window x batch x dim
            m (T.tensor): mask. window x batch. For masking in recurrent steps.
            h0s (Optional[list]): initial h0s.
            condition_on (Optional[T.tensor]): conditional for recurrent step.

        Returns:
            OrderedDict: dictionary of results: hiddens, probabilities, and
                preacts.
            theano.OrderedUpdates.

        '''
        constants = []

        if h0s is None and self.init_net is not None:
            h0s = self.init_net.initialize(x[0])
            constants += h0s
        elif h0s is None:
            h0s = [T.alloc(0., x.shape[1], dim_h).astype(floatX) for dim_h in self.dim_hs]

        if m is None:
            m = T.ones((x.shape[0], x.shape[1])).astype(floatX)

        params = self.get_sample_params()

        results, updates = self.step_call(x, m, h0s, *params)
        results['h0s'] = h0s
        return results, updates, constants

    def sample(self, x0=None, h0s=None, n_samples=10, n_steps=10,
               condition_on=None, debug=False):
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
                p=T.constant(0.5).astype(floatX),
                size=(n_samples, self.output_net.dim_out)).astype(floatX)

        if h0s is None and self.init_net is not None:
            h0s = self.init_net.initialize(x0)
        elif h0s is None:
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
    def factory(dim_in=None, dim_out=None, dim_h=None, **kwargs):
        '''Convenience factory for SimpleRNN (see `RNN.factory`).

        '''
        if dim_out is None:
            dim_out = dim_in

        mlps, kwargs = RNN.mlp_factory(dim_in, dim_out, [dim_h], **kwargs)
        kwargs.update(**mlps)

        return SimpleRNN(dim_in, dim_h, dim_out=dim_out, **kwargs)

    def energy(self, X, h0=None):
        '''Energy function.

        '''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).energy(X, h0s=h0s)

    def __call__(self, x, m=None, h0=None, condition_on=None):
        '''Call function (see `RNN.__call__`).

        '''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).__call__(
            x, m=m, h0s=h0s, condition_on=condition_on)

    def sample(self, x0=None, h0=None, **kwargs):
        '''Sample the SimpleRNN (see `RNN.sample`).

        '''
        if h0 is not None:
            h0s = [h0]
        else:
            h0s = None
        return super(SimpleRNN, self).sample(x0=x0, h0s=h0s, **kwargs)
