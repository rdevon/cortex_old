'''Convolutional NN

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from . import resolve_nonlinearity
from .. import batch_normalization, Cell, dropout, norm_weight
from ...utils import floatX


class CNN2D(Cell):
    _required = ['input_shape', 'n_filters', 'filter_shapes', 'pool_sizes']
    _options = {'dropout': False, 'weight_noise': 0,
                'batch_normalization': False}
    _args = ['input_shape', 'n_filters', 'filter_shapes', 'pool_sizes',
             'h_act', 'out_act']
    _dim_map = {'output': None}

    def __init__(self, input_shape, n_filters, filter_shapes, pool_sizes,
                 h_act='sigmoid', out_act=None, name='CNN2D', **kwargs):
        if not(len(n_filters) == len(filter_shapes)):
            raise TypeError(
            '`filter_shapes` and `n_filters` must have the same length')

        if out_act is None: out_act = h_act
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.n_filters = n_filters
        self.n_layers = len(self.n_filters)
        self.pool_sizes = pool_sizes
        self.h_act = resolve_nonlinearity(h_act)
        self.out_act = resolve_nonlinearity(out_act)

        super(CNN2D, self).__init__(name=name, **kwargs)

    @classmethod
    def set_link_value(C, key, input_shape=None, filter_shapes=None,
                       n_filters=None, pool_sizes=None, **kwargs):

        if key not in ['output']:
            return super(CNN2D, C).set_link_value(link, key)
        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pool_sizes is None: raise ValueError('pool_sizes')
        input_shape = input_shape[1:]

        for filter_shape, pool_size in zip(filter_shapes, pool_sizes):
            dim_x = (input_shape[0] - filter_shape[0] + 1) // pool_size[0]
            dim_y = (input_shape[1] - filter_shape[1] + 1) // pool_size[1]
            input_shape = (dim_x, dim_y)

        return dim_x * dim_y * n_filters[-1]

    def init_params(self, weight_scale=1e-3, dim_in=None):
        dim_ins = [self.input_shape[0]] + self.n_filters[:-1]
        dim_outs = self.n_filters

        weights = []
        biases = []

        for dim_in, dim_out, pool_size, (dim_x, dim_y) in zip(
            dim_ins, dim_outs, self.pool_sizes, self.filter_shapes):
            fan_in = dim_in * dim_x * dim_y
            fan_out = dim_out * dim_x * dim_y // np.prod(pool_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = self.rng.uniform(low=-W_bound, high=W_bound,
                                 size=(dim_out, dim_in, dim_x, dim_y))
            b = np.zeros((dim_out,))
            weights.append(W)
            biases.append(b)

        self.params = dict(weights=weights, biases=biases)

    def get_params(self):
        params = zip(self.weights, self.biases)
        params = [i for sl in params for i in sl]
        return super(CNN2D, self).get_params(params=params)

    def init_args(self, X, batch_size=None):
        if batch_size is None:
            session = self.manager._current_session
            batch_size = session.batch_size

        return (X, batch_size)

    def _feed(self, X, batch_size, *params):
        session = self.manager._current_session
        params = list(params)
        outs = OrderedDict(X=X)
        outs['input'] = X

        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1],
                       self.input_shape[2]))
        input_shape = (batch_size,) + self.input_shape

        for l in xrange(self.n_layers):
            if self.batch_normalization:
                self.logger.debug('Batch normalization on layer %d' % l)
                X = batch_normalization(X, session=session)

            W = params.pop(0)
            b = params.pop(0)

            shape = self.filter_shapes[l]
            pool_size = self.pool_sizes[l]
            n_filters = self.n_filters[l]
            filter_shape = (n_filters, input_shape[1]) + shape

            conv_out = T.nnet.conv2d(input=X, filters=W,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape)

            dim_x = input_shape[2] - shape[0] + 1
            dim_y = input_shape[3] - shape[1] + 1

            pool_out = T.signal.pool.pool_2d(input=conv_out, ds=pool_size,
                                             ignore_border=True)

            dim_x = dim_x // pool_size[0]
            dim_y = dim_y // pool_size[1]

            outs.update(**{
                ('C_%d' % l): conv_out,
                ('P_%d' % l): pool_out})

            preact = pool_out + b[None, :, None, None]

            if l < self.n_layers - 1:
                X = self.h_act(preact)

                if self.dropout and self.noise_switch():
                    self.logger.debug('Adding dropout to layer {layer} for CNN2D '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = dropout(X, self.h_act, self.dropout, self.trng)

                outs.update(**{
                    ('G_%d' % l): preact,
                    ('H_%d' % l): X})

            else:
                X = self.out_act(preact)
                outs['Z'] = preact
                outs['Y'] = X

            input_shape = (batch_size, filter_shape[0], dim_x, dim_y)

        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        outs['output'] = X

        assert len(params) == 0
        return outs


class RCNN2D(CNN2D):
    _required = ['output_shape', 'n_filters', 'filter_shapes', 'pool_sizes']
    _options = {'dropout': False, 'weight_noise': 0,
                'batch_normalization': False}
    _args = ['output_shape', 'n_filters', 'filter_shapes', 'pool_sizes',
             'h_act', 'out_act']
    _dim_map = {'input': dim_in}

    def __init__(self, input_shape, n_filters, filter_shapes, pool_sizes,
                 h_act='sigmoid', out_act=None, name='CNN2D', **kwargs):
        if not(len(n_filters) == len(filter_shapes)):
            raise TypeError(
            '`filter_shapes` and `n_filters` must have the same length')

        if out_act is None: out_act = h_act
        self.input_shape = input_shape
        self.filter_shapes = filter_shapes
        self.n_filters = n_filters
        self.n_layers = len(self.n_filters)
        self.pool_sizes = pool_sizes
        self.h_act = resolve_nonlinearity(h_act)
        self.out_act = resolve_nonlinearity(out_act)

        super(CNN2D, self).__init__(name=name, **kwargs)

    @classmethod
    def set_link_value(C, key, input_shape=None, filter_shapes=None,
                       n_filters=None, pool_sizes=None, **kwargs):

        if key not in ['output']:
            return super(CNN2D, C).set_link_value(link, key)
        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pool_sizes is None: raise ValueError('pool_sizes')
        input_shape = input_shape[1:]

        for filter_shape, pool_size in zip(filter_shapes, pool_sizes):
            dim_x = (input_shape[0] - filter_shape[0] + 1) // pool_size[0]
            dim_y = (input_shape[1] - filter_shape[1] + 1) // pool_size[1]
            input_shape = (dim_x, dim_y)

        return dim_x * dim_y * n_filters[-1]

    def init_params(self, weight_scale=1e-3, dim_in=None):
        dim_ins = [self.input_shape[0]] + self.n_filters[:-1]
        dim_outs = self.n_filters

        weights = []
        biases = []

        for dim_in, dim_out, pool_size, (dim_x, dim_y) in zip(
            dim_ins, dim_outs, self.pool_sizes, self.filter_shapes):
            fan_in = dim_in * dim_x * dim_y
            fan_out = dim_out * dim_x * dim_y // np.prod(pool_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = self.rng.uniform(low=-W_bound, high=W_bound,
                                 size=(dim_out, dim_in, dim_x, dim_y))
            b = np.zeros((dim_out,))
            weights.append(W)
            biases.append(b)

        self.params = dict(weights=weights, biases=biases)

    def _feed(self, X, batch_size, *params):
        session = self.manager._current_session
        params = list(params)
        outs = OrderedDict(X=X)
        outs['input'] = X

        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1],
                       self.input_shape[2]))
        input_shape = (batch_size,) + self.input_shape

        for l in xrange(self.n_layers):
            if self.batch_normalization:
                self.logger.debug('Batch normalization on layer %d' % l)
                X = batch_normalization(X, session=session)

            W = params.pop(0)
            b = params.pop(0)

            shape = self.filter_shapes[l]
            pool_size = self.pool_sizes[l]
            n_filters = self.n_filters[l]
            filter_shape = (n_filters, input_shape[1]) + shape

            conv_out = T.nnet.conv2d(input=X, filters=W,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape)

            dim_x = input_shape[2] - shape[0] + 1
            dim_y = input_shape[3] - shape[1] + 1

            pool_out = T.signal.pool.pool_2d(input=conv_out, ds=pool_size,
                                             ignore_border=True)

            dim_x = dim_x // pool_size[0]
            dim_y = dim_y // pool_size[1]

            outs.update(**{
                ('C_%d' % l): conv_out,
                ('P_%d' % l): pool_out})

            preact = pool_out + b[None, :, None, None]

            if l < self.n_layers - 1:
                X = self.h_act(preact)

                if self.dropout and self.noise_switch():
                    self.logger.debug('Adding dropout to layer {layer} for CNN2D '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = dropout(X, self.h_act, self.dropout, self.trng)

                outs.update(**{
                    ('G_%d' % l): preact,
                    ('H_%d' % l): X})

            else:
                X = self.out_act(preact)
                outs['Z'] = preact
                outs['Y'] = X

            input_shape = (batch_size, filter_shape[0], dim_x, dim_y)

        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        outs['output'] = X

        assert len(params) == 0
        return outs

_classes = {'CNN2D': CNN2D}