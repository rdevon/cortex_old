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
             'h_act']
    _dim_map = {'output': None}
    _components = {'fully_connected': None}

    def __init__(self, input_shape, n_filters, filter_shapes, pool_sizes,
                 h_act='sigmoid', out_act=None, name='CNN2D', border_modes=None,
                 dim_hs=None, dim_out=None, **kwargs):
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
        if out_act is None: out_act = h_act
        self.out_act = resolve_nonlinearity(out_act)
        self.border_modes = border_modes or (['full_and_trim'] * self.n_layers)
        
        super(CNN2D, self).__init__(name=name, dim_hs=dim_hs,
                                    dim_out=dim_out, h_act=h_act,
                                    out_act=out_act, **kwargs)
    
    def set_components(self, components=None, dim_hs=None, dim_in=None,
                       dim_out=None, h_act=None, out_act=None, **kwargs):

        components = OrderedDict((k, v) for k, v in self._components.items())
        if dim_out is not None:
            dim_in = self.set_link_value(
                'output', input_shape=self.input_shape,
                filter_shapes=self.filter_shapes, n_filters=self.n_filters,
                pool_sizes=self.pool_sizes, border_modes=self.border_modes)
            dim_hs = dim_hs or []
            components['fully_connected'] = OrderedDict(
                cell_type='MLP',
                dim_hs=dim_hs,
                h_act=h_act,
                out_act=out_act,
                dim_in=dim_in,
                dim_out=dim_out,
                dropout=kwargs.get('dropout', False),
                batch_normalization=self.batch_normalization)

        return super(CNN2D, self).set_components(components=components, **kwargs)
    
    @classmethod
    def set_link_value(C, key, input_shape=None, filter_shapes=None,
                       n_filters=None, pool_sizes=None, border_modes=None,
                       **kwargs):

        if key not in ['output']:
            return super(CNN2D, C).set_link_value(link, key)

        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pool_sizes is None: raise ValueError('pool_sizes')
        n_layers = len(n_filters)
        if border_modes is None: border_modes = ['full'] * n_layers
        if not(n_layers == len(filter_shapes) == len(pool_sizes) ==
               len(border_modes)):
            raise TypeError('All list inputs must have the same length')

        input_shape = input_shape[1:]

        for filter_shape, pool_size, border_mode in zip(
            filter_shapes, pool_sizes, border_modes):
            if border_mode == 'valid':
                dim_x = input_shape[0] - filter_shape[0] + 1
                dim_y = input_shape[1] - filter_shape[1] + 1
            elif border_mode == 'full':
                dim_x = input_shape[0] + filter_shape[0] - 1
                dim_y = input_shape[1] + filter_shape[1] - 1
            elif border_mode == 'full_and_trim':
                dim_x, dim_y = input_shape[:2]
            elif border_mode == 'half':
                dim_x = input_shape[0] + ((filter_shape[0] + 1) % 2)
                dim_y = input_shape[1] + ((filter_shape[1] + 1) % 2)
            else:
                raise NotImplementedError(border_mode)

            dim_x = dim_x // pool_size[0]
            dim_y = dim_y // pool_size[1]
            input_shape = (dim_x, dim_y)

        return dim_x * dim_y * n_filters[-1]

    def init_params(self, weight_scale=1e-3, dim_in=None):
        self.params = OrderedDict()
        dim_ins = [self.input_shape[0]] + self.n_filters[:-1]
        dim_outs = self.n_filters

        weights = []
        biases = []
        if self.batch_normalization:
            gammas = []
            betas = []
            
        self.trims = []

        im_x, im_y = self.input_shape[1:]
        for i in xrange(self.n_layers):
            dim_in = dim_ins[i]
            dim_out = dim_outs[i]
            pool_size = self.pool_sizes[i]
            dim_x, dim_y = self.filter_shapes[i]
            border_mode = self.border_modes[i]
            
            fan_in = dim_in * dim_x * dim_y
            fan_out = dim_out * dim_x * dim_y // np.prod(pool_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = self.rng.uniform(low=-W_bound, high=W_bound,
                                 size=(dim_out, dim_in, dim_x, dim_y))
            self.trims.append((int(np.floor(W.shape[2] / 2.)),
                               int(np.floor(W.shape[3] / 2.))))
            b = np.zeros((dim_out,))
            weights.append(W)
            biases.append(b)
            if self.batch_normalization:
                bn_shape = (dim_in * im_x * im_y,)
                gamma = np.ones(bn_shape)
                beta = np.zeros(bn_shape)
                gammas.append(gamma)
                betas.append(beta)
            im_x = im_x // pool_size[0]
            im_y = im_y // pool_size[1]

        self.params['weights'] = weights
        self.params['biases'] = biases
        if self.batch_normalization:
           self.params['gammas'] = gammas
           self.params['betas'] = betas

    def get_params(self):
        if self.batch_normalization:
            params = zip(self.weights, self.biases, self.gammas, self.betas)
        else:
            params = zip(self.weights, self.biases)
        params = [i for sl in params for i in sl]
        return super(CNN2D, self).get_params(params=params)

    def init_args(self, X, batch_size=None):
        if batch_size is None:
            session = self.manager._current_session
            batch_size = session.batch_size

        return (X, batch_size)

    def feed(self, X, batch_size, *params):
        print X, batch_size, params
        session = self.manager._current_session
        params = list(params)

        if X.ndim == 3:
            reshape = X.shape
            X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        else:
            reshape = None

        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1],
                       self.input_shape[2]))
        input_shape = (batch_size,) + self.input_shape
        outs = OrderedDict(X=X)
        outs['input'] = X

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)
            if self.batch_normalization:
                gamma = params.pop(0)
                beta = params.pop(0)
                self.logger.debug('Batch normalization on layer %d' % l)
                shape = X.shape
                X = batch_normalization(
                    X.reshape((shape[0], shape[1] * shape[2] * shape[3])),
                    gamma, beta, session=session)
                X = X.reshape(shape)

            shape = self.filter_shapes[l]
            pool_size = self.pool_sizes[l]
            n_filters = self.n_filters[l]
            border_mode = self.border_modes[l]
            if border_mode == 'full_and_trim':
                _border_mode = 'full'
                trim = True
            else:
                _border_mode = border_mode
                trim = False
                
            filter_shape = (n_filters, input_shape[1]) + shape
            
            outs.update(**{'X_%d' % l: X})
            
            conv_out = T.nnet.conv2d(input=X, filters=W,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape,
                                     border_mode=_border_mode)
            
            outs.update(**{'Ci_%d' % l: conv_out})
            
            if trim:
                sx, sy = self.trims[l]
                conv_out = conv_out[:, :, sx:-sx, sy:-sy]
            
            if border_mode == 'valid':
                dim_x = input_shape[2] - shape[0] + 1
                dim_y = input_shape[3] - shape[1] + 1
            elif border_mode == 'full':
                dim_x = input_shape[2] + shape[0] - 1
                dim_y = input_shape[3] + shape[1] - 1
            elif border_mode == 'full_and_trim':
                dim_x, dim_y = input_shape[2:]
            elif border_mode == 'half':
                dim_x = input_shape[2] + ((shape[0] + 1) % 2)
                dim_y = input_shape[3] + ((shape[1] + 1) % 2)
            else:
                raise NotImplementedError(border_mode)

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
                    epsilon = params.pop(0)
                    self.logger.debug('Adding dropout to layer {layer} for MLP '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = dropout(X, self.h_act, self.dropout, self.trng,
                                epsilon=epsilon)

                outs.update(**{
                    ('G_%d' % l): preact,
                    ('H_%d' % l): X})

            else:
                X = self.out_act(preact)
                outs['Z'] = preact
                outs['Y'] = X

            input_shape = (batch_size, filter_shape[0], dim_x, dim_y)
            
        outs['Xo'] = X
        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        outs['Xr'] = X
        
        if self.fully_connected is not None:
            ffn_outs = self.fully_connected._feed(X, *params)
            outs.update(('ffn_{}'.format(k), v) for k, v in ffn_outs.items())
            X = ffn_outs['Y']

        if reshape is not None:
            X = X.reshape((reshape[0], reshape[1], X.shape[1]))
        outs['output'] = X

        return outs
    
    def _feed(self, X, batch_size, *params):
        params = list(params)
        if self.dropout and self.noise_switch():
            epsilons = self.dropout_epsilons(
                size=(X.shape[0],), input_shape=(X.shape[2], X.shape[3]))
            params = self.get_epsilon_params(epsilons, *params)

        return self.feed(X, batch_size, *params)

    def get_epsilon_params(self, epsilons, *params):
        if self.dropout and self.noise_switch():
            if self.batch_normalization:
                ppl = 4
            else:
                ppl = 2
            new_params = []
            for l in xrange(self.n_layers - 1):
                new_params += params[ppl*l:ppl*(l+1)]
                new_params.append(epsilons[l])
            new_params += params[ppl*(self.n_layers-1):]
            assert len(new_params) == len(params) + self.n_layers - 1

            return new_params
        else:
            return params

    def dropout_epsilons(self, size, input_shape):
        epsilons = []
        
        for filter_shape, pool_size, border_mode, n_filter in zip(
            self.filter_shapes, self.pool_sizes, self.border_modes, self.n_filters):
            if border_mode == 'valid':
                dim_x = input_shape[0] - filter_shape[0] + 1
                dim_y = input_shape[1] - filter_shape[1] + 1
            elif border_mode == 'full':
                dim_x = input_shape[0] + filter_shape[0] - 1
                dim_y = input_shape[1] + filter_shape[1] - 1
            elif border_mode == 'half':
                dim_x = input_shape[0] + ((filter_shape[0] + 1) % 2)
                dim_y = input_shape[1] + ((filter_shape[1] + 1) % 2)
            else:
                raise NotImplementedError(border_mode)

            dim_x = dim_x // pool_size[0]
            dim_y = dim_y // pool_size[1]
            input_shape = (dim_x, dim_y)
            shape = size + (n_filter, dim_x, dim_y)
            eps = self.trng.binomial(shape, p=1-self.dropout, n=1, dtype=floatX)
            epsilons.append(eps)

        return epsilons


class RCNN2D(CNN2D):
    def __init__(self, input_shape, n_filters, filter_shapes, pool_sizes,
                 border_modes=None, dim_in=None, name='RCNN2D', **kwargs):
        n_layers = len(n_filters)
        border_modes = border_modes or (['full_and_trim'] * n_layers)
        super(RCNN2D, self).__init__(input_shape, n_filters, filter_shapes,
                                     pool_sizes, border_modes=border_modes,
                                     dim_in=dim_in, name=name, **kwargs)
        
    def set_components(self, components=None, dim_hs=None, dim_in=None,
                       dim_out=None, h_act=None, out_act=None, **kwargs):

        components = OrderedDict((k, v) for k, v in self._components.items())
        if dim_in is not None:
            dim_out = reduce(lambda x, y: x * y, self.input_shape)
            dim_hs = dim_hs or []
            components['fully_connected'] = OrderedDict(
                cell_type='MLP',
                dim_hs=dim_hs,
                h_act=h_act,
                out_act=out_act,
                dim_in=dim_in,
                dim_out=dim_out,
                dropout=kwargs.get('dropout', False),
                batch_normalization=self.batch_normalization)

        return super(CNN2D, self).set_components(components=components, **kwargs)

    @classmethod
    def set_link_value(C, key, input_shape=None, filter_shapes=None,
                       n_filters=None, pool_sizes=None, border_modes=None,
                       **kwargs):

        if key not in ['output']:
            return super(RCNN2D, C).set_link_value(link, key)

        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pool_sizes is None: raise ValueError('pool_sizes')
        n_layers = len(n_filters)
        if border_modes is None: border_modes = ['full'] * n_layers
        if not(n_layers == len(filter_shapes) == len(pool_sizes) ==
               len(border_modes)):
            raise TypeError('All list inputs must have the same length')

        input_shape = input_shape[1:]

        for filter_shape, pool_size, border_mode in zip(
            filter_shapes, pool_sizes, border_modes):

            dim_x = input_shape[0] * pool_size[0]
            dim_y = input_shape[1] * pool_size[1]

            if border_mode == 'valid':
                dim_x = dim_x - filter_shape[0] + 1
                dim_y = dim_y - filter_shape[1] + 1
            elif border_mode == 'full':
                dim_x = dim_x + filter_shape[0] - 1
                dim_y = dim_y + filter_shape[1] - 1
            elif border_mode == 'full_and_trim':
                dim_x, dim_y = input_shape[:2]
            elif border_mode == 'half':
                dim_x = dim_x + ((filter_shape[0] + 1) % 2)
                dim_y = dim_y + ((filter_shape[1] + 1) % 2)
            else:
                raise NotImplementedError(border_mode)
            input_shape = (dim_x, dim_y)

        return dim_x * dim_y * n_filters[-1]
    
    def init_params(self, weight_scale=1e-3, dim_in=None):
        self.params = OrderedDict()
        dim_ins = [self.input_shape[0]] + self.n_filters[:-1]
        dim_outs = self.n_filters

        weights = []
        biases = []
        if self.batch_normalization:
            gammas = []
            betas = []
            
        self.trims = []

        im_x, im_y = self.input_shape[1:]
        for i in xrange(self.n_layers):
            dim_in = dim_ins[i]
            dim_out = dim_outs[i]
            pool_size = self.pool_sizes[i]
            dim_x, dim_y = self.filter_shapes[i]
            border_mode = self.border_modes[i]
            
            fan_in = dim_in * dim_x * dim_y
            fan_out = dim_out * dim_x * dim_y // np.prod(pool_size)
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = self.rng.uniform(low=-W_bound, high=W_bound,
                                 size=(dim_out, dim_in, dim_x, dim_y))
            self.trims.append((int(np.floor(W.shape[2] / 2.)),
                               int(np.floor(W.shape[3] / 2.))))
            b = np.zeros((dim_out,))
            weights.append(W)
            biases.append(b)
            if self.batch_normalization:
                bn_shape = (dim_in * im_x * im_y,)
                gamma = np.ones(bn_shape)
                beta = np.zeros(bn_shape)
                gammas.append(gamma)
                betas.append(beta)
            im_x = im_x * pool_size[0]
            im_y = im_y * pool_size[1]

        self.params['weights'] = weights
        self.params['biases'] = biases
        if self.batch_normalization:
           self.params['gammas'] = gammas
           self.params['betas'] = betas

    def feed(self, X, batch_size, *params):
        session = self.manager._current_session
        params = list(params)
        
        if X.ndim == 2:
            reshape = None
        elif X.ndim == 3:
            reshape = (X.shape[0], X.shape[1])
            X = X.reshape((X.shape[0] * X.shape[1], X.shape[2]))
        else:
            raise ValueError()
        
        outs = OrderedDict()
        if self.fully_connected is not None:
            ffn_params = self.select_params('fully_connected', *params)
            params = params[:-len(ffn_params)]
            ffn_outs = self.fully_connected._feed(X, *ffn_params)
            outs.update(('ffn_{}'.format(k), v) for k, v in ffn_outs.items())
            X = ffn_outs['Y']

        X = X.reshape((X.shape[0], self.input_shape[0], self.input_shape[1],
                       self.input_shape[2]))
        input_shape = (batch_size,) + self.input_shape
        outs.update(X=X, input=X)

        def depool(X, pool, out, l):
            '''
            From https://gist.github.com/kastnerkyle/
            '''
            shape = (X.shape[1], X.shape[2] * pool[0], X.shape[3] * pool[1])
            stride = X.shape[2]
            offset = X.shape[3]
            dim_in = stride * offset
            dim_out = dim_in * pool[0] * pool[1]
            upsampled = T.zeros((dim_in, dim_out))

            rs = T.arange(dim_in)
            cs = rs * pool[1] + (rs // stride * pool[0] * offset)
            upsampled = T.set_subtensor(upsampled[rs, cs], 1.)
            X_f = X.reshape((X.shape[0], shape[0], X.shape[2] * X.shape[3]))

            upsampled_f = T.dot(X_f, upsampled)
            upsampled = upsampled_f.reshape(
                (X.shape[0], shape[0], shape[1], shape[2]))

            return upsampled

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)
            
            if self.batch_normalization:
                gamma = params.pop(0)
                beta = params.pop(0)
                self.logger.debug('Batch normalization on layer %d' % l)
                shape = X.shape
                X = batch_normalization(
                    X.reshape((shape[0], shape[1] * shape[2] * shape[3])),
                    gamma, beta, session=session)
                X = X.reshape(shape)

            shape = self.filter_shapes[l]
            pool_size = self.pool_sizes[l]
            n_filters = self.n_filters[l]
            border_mode = self.border_modes[l]

            filter_shape = (n_filters, input_shape[1]) + shape

            if pool_size != (1, 1):
                unpool_out = depool(X, pool_size, outs, l)
            else:
                unpool_out = X

            dim_x = input_shape[2] * pool_size[0]
            dim_y = input_shape[3] * pool_size[1]
            input_shape = (input_shape[0], input_shape[1], dim_x, dim_y)
            
            if border_mode == 'full_and_trim':
                _border_mode = 'full'
                trim = True
            else:
                _border_mode = border_mode
                trim = False

            conv_out = T.nnet.conv2d(input=unpool_out, filters=W,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape,
                                     border_mode=_border_mode)
            
            if trim:
                sx, sy = self.trims[l]
                conv_out = conv_out[:, :, sx:-sx, sy:-sy]

            if border_mode == 'valid':
                dim_x = dim_x - shape[0] + 1
                dim_y = dim_y - shape[1] + 1
            elif border_mode == 'full':
                dim_x = dim_x + shape[0] - 1
                dim_y = dim_y + shape[1] - 1
            elif border_mode == 'full_and_trim':
                dim_x, dim_y = input_shape[2:]
            elif border_mode == 'half':
                dim_x = dim_x + ((shape[0] + 1) % 2)
                dim_y = dim_y + ((shape[1] + 1) % 2)
            else:
                raise NotImplementedError(border_mode)

            outs['P_%d' % l] = unpool_out
            outs['C_%d' % l] = conv_out

            preact = conv_out + b[None, :, None, None]

            if l < self.n_layers - 1:
                X = self.h_act(preact)

                if self.dropout and self.noise_switch():
                    epsilon = params.pop(0)
                    self.logger.debug('Adding dropout to layer {layer} for MLP '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = dropout(X, self.h_act, self.dropout, self.trng,
                                epsilon=epsilon)

                outs.update(**{
                    ('G_%d' % l): preact,
                    ('H_%d' % l): X})

            else:
                X = self.out_act(preact)
                outs['Z'] = preact
                outs['Y'] = X

            input_shape = (batch_size, filter_shape[0], dim_x, dim_y)

        X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
        if reshape is not None:
            X = X.reshape((reshape[0], reshape[1], X.shape[1]))
        outs['output'] = X

        return outs
    

_classes = {'CNN2D': CNN2D, 'RCNN2D': RCNN2D}