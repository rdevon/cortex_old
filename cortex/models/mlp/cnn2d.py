'''Convolutional NN

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T
from theano.tensor.nnet.bn import batch_normalization

from . import resolve_nonlinearity
from .. import Cell, dropout, norm_weight
from ...utils import floatX


def get_output_shape(input_shape, filter_shape, stride, pool=1, pad=0):
    if pad == 'valid':
        output_shape = input_shape - filter_shape + 1
    elif pad == 'full':
        output_shape = input_shape + filter_shape - 1
    elif pad == 'full_and_trim':
        output_shape = input_shape
    elif pad == 'half':
        output_shape = input_shape + ((filter_shape + 1) % 2)
    elif isinstance(pad, int):
        output_shape = input_shape + 2 * pad - filter_shape + 1
    else:
        raise NotImplementedError(pad)

    output_shape = (output_shape + stride - 1) // stride
    output_shape = output_shape // pool
    
    return output_shape

def get_input_shape(output_shape, filter_shape, stride, pool=1, pad=0):
    output_shape = output_shape * pool

    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'full_and_trim':
        return (output_shape - 1) * stride
    elif isinstance(pad, int):
        pass
    else:
        raise NotImplementedError(pad)
    
    input_shape = (output_shape - 1) * stride - 2 * pad + filter_shape
    
    return input_shape


class CNN2D(Cell):
    _required = ['input_shape', 'n_filters', 'filter_shapes']
    _options = {'dropout': False, 'weight_noise': 0,
                'batch_normalization': False}
    _args = ['input_shape', 'n_filters', 'filter_shapes', 'pools', 'pads',
             'strides', 'h_act']
    _dim_map = {'output': None}
    _components = {'fully_connected': None}

    def __init__(self,
                 input_shape, n_filters, filter_shapes,
                 pools=None, strides=None, pads=None,
                 h_act='sigmoid', out_act=None, 
                 dim_hs=None, dim_out=None,
                 name='CNN2D', **kwargs):
        if not(len(n_filters) == len(filter_shapes)):
            raise TypeError(
            '`filter_shapes` and `n_filters` must have the same length')

        if out_act is None: out_act = h_act
        self.n_layers = len(n_filters)
        self.h_act = resolve_nonlinearity(h_act)
        if out_act is None: out_act = h_act
        self.out_act = resolve_nonlinearity(out_act)

        self.input_shape = input_shape
        
        self.filter_shapes = filter_shapes
        self.n_filters = n_filters
        
        self.pools = pools or [(1, 1) for _ in range(self.n_layers)]
        self.strides = strides or [(1, 1) for _ in range(self.n_layers)]
        self.pads = pads or ['full_and_trim' for _ in range(self.n_layers)]
        
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
                pools=self.pools, strides=self.strides, pads=self.pads)
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
                       n_filters=None, pools=None, strides=None,
                       pads=None, **kwargs):

        if key not in ['output']:
            return super(CNN2D, C).set_link_value(link, key)

        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pools is None: raise ValueError('pools')
        if pads is None: raise ValueError('pads')
        if strides is None: raise ValueError('strides')
        
        n_layers = len(n_filters)
        if not(n_layers == len(filter_shapes) == len(pools) ==
               len(strides) == len(pads)):
            raise TypeError('All list inputs must have the same length')

        dim_x, dim_y = input_shape[1:]

        for filter_shape, pool, stride, pad in zip(
            filter_shapes, pools, strides, pads):
            if isinstance(pad, str): pad = (pad, pad)
            dim_x, dim_y = tuple(get_output_shape(i, f, s, p, pa)
                                 for i, f, s, p, pa
                                 in zip((dim_x, dim_y), filter_shape, stride, pool,
                                        pad))

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
            pool = self.pools[i]
            dim_x, dim_y = self.filter_shapes[i]
            stride = self.strides[i]
            pad = self.pads[i]
            
            fan_in = dim_in * dim_x * dim_y
            fan_out = dim_out * dim_x * dim_y // np.prod(pool)
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
                
            if isinstance(pad, str): pad = (pad, pad)
            im_x, im_y = tuple(get_output_shape(i, f, s, p, pa)
                               for i, f, s, p, pa
                               in zip((im_x, im_y), (dim_x, dim_y), stride,
                                      pool, pad))

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

        dim_x = input_shape[2]
        dim_y = input_shape[3]
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)
            if self.batch_normalization:
                gamma = params.pop(0)
                beta = params.pop(0)
                self.logger.debug('Batch normalization on layer %d' % l)
                shape = X.shape
                X = X.reshape((shape[0], shape[1] * shape[2] * shape[3]))
                mean = X.mean(0, keepdims=True)
                std = X.std(0, keepdims=True)
                X = batch_normalization(inputs=X, gamma=gamma, beta=beta,
                                        mean=mean, std=std)
                X = X.reshape(shape)

            shape = self.filter_shapes[l]
            pool = self.pools[l]
            n_filters = self.n_filters[l]
            pad = self.pads[l]
            stride = self.strides[l]
            
            if pad == 'full_and_trim':
                _pad = 'full'
                trim = True
            else:
                _pad = pad
                trim = False
                
            filter_shape = (n_filters, input_shape[1]) + shape
            
            outs.update(**{'X_%d' % l: X})
            
            conv_out = T.nnet.conv2d(input=X, filters=W,
                                     filter_shape=filter_shape,
                                     input_shape=input_shape,
                                     subsample=stride,
                                     border_mode=_pad,
                                     filter_flip=True)
            
            outs.update(**{'Ci_%d' % l: conv_out})
            
            if trim:
                sx, sy = self.trims[l]
                conv_out = conv_out[:, :, sx:-sx, sy:-sy]

            if pool != (1, 1):
                pool_out = T.signal.pool.pool_2d(input=conv_out, ds=pool,
                                                 ignore_border=True)
            else:
                pool_out = conv_out

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
            
            if isinstance(_pad, str): _pad = (_pad, _pad)
            dim_x, dim_y = tuple(get_output_shape(i, f, s, p, pa)
                                 for i, f, s, p, pa
                                 in zip([dim_x, dim_y], shape, stride,
                                    pool, _pad))

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
        
        for filter_shape, pool, pad, stride, n_filter in zip(
            self.filter_shapes, self.pools, self.pads, self.strides, self.n_filters):
            
            if isinstance(pad, str): pad = (pad, pad)
            dim_x, dim_y = tuple(get_output_shape(i, f, s, p, pa)
                                 for i, f, s, p, pa
                                 in zip(input_shape[1:], filter_shape, stride, pool,
                                        pad))
            
            input_shape = (dim_x, dim_y)
            shape = size + (n_filter, dim_x, dim_y)
            eps = self.trng.binomial(shape, p=1-self.dropout, n=1, dtype=floatX)
            epsilons.append(eps)

        return epsilons


class RCNN2D(CNN2D):
    def __init__(self,
                 input_shape, n_filters, filter_shapes,
                 dim_in=None, name='RCNN2D', **kwargs):
        super(RCNN2D, self).__init__(input_shape, n_filters, filter_shapes,
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
                       n_filters=None, pools=None, strides=None,
                       pads=None, **kwargs):

        if key not in ['output']:
            return super(RCNN2D, C).set_link_value(link, key)

        if n_filters is None: raise ValueError('n_filters')
        if input_shape is None: raise ValueError('input_shape')
        if filter_shapes is None: raise ValueError('filter_shapes')
        if pools is None: raise ValueError('pools')
        if pads is None: raise ValueError('pads')
        if strides is None: raise ValueError('strides')
        
        n_layers = len(n_filters)
        if not(n_layers == len(filter_shapes) == len(pools) ==
               len(strides) == len(pads)):
            raise TypeError('All list inputs must have the same length')

        input_shape = input_shape[1:]

        for filter_shape, pool, stride, pad in zip(
            filter_shapes, pools, strides, pads):
            if isinstance(pad, str): pad = (pad, pad)
            dim_x, dim_y = tuple(get_input_shape(i, f, s, p, pa)
                                 for i, f, s, p, pa
                                 in zip(input_shape, filter_shape, stride, pool,
                                        pad))
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
            pool = self.pools[i]
            dim_x, dim_y = self.filter_shapes[i]
            stride = self.strides[i]
            pad = self.pads[i]
            
            fan_in = dim_in * dim_x * dim_y
            fan_out = dim_out * dim_x * dim_y // np.prod(pool)
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
                
            if isinstance(pad, str): pad = (pad, pad)
            im_x, im_y = tuple(get_input_shape(i, f, s, p, pa)
                               for i, f, s, p, pa
                               in zip([im_x, im_y], [dim_x, dim_y], stride,
                                      pool, pad))

        self.params['weights'] = weights
        self.params['biases'] = biases
        if self.batch_normalization:
           self.params['gammas'] = gammas
           self.params['betas'] = betas

    def feed(self, X, batch_size, *params):
        session = self.manager._current_session
        params = list(params)
        assert batch_size is not None
        
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

        dim_x = input_shape[2]
        dim_y = input_shape[3]
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)
            
            if self.batch_normalization:
                gamma = params.pop(0)
                beta = params.pop(0)
                self.logger.debug('Batch normalization on layer %d' % l)
                shape = X.shape
                X = X.reshape((shape[0], shape[1] * shape[2] * shape[3]))
                mean = X.mean(0, keepdims=True)
                std = X.std(0, keepdims=True)
                X = batch_normalization(inputs=X, gamma=gamma, beta=beta,
                                        mean=mean, std=std)
                X = X.reshape(shape)

            shape = self.filter_shapes[l]
            pool = self.pools[l]
            n_filters = self.n_filters[l]
            pad = self.pads[l]
            stride = self.strides[l]

            filter_shape = (n_filters, input_shape[1]) + shape

            if pool != (1, 1):
                unpool_out = depool(X, pool, outs, l)
            else:
                unpool_out = X

            input_shape = (input_shape[0], input_shape[1],
                           input_shape[2] * pool[0], input_shape[3] * pool[1])
            
            if pad == 'full_and_trim':
                _pad = 'full'
                trim = True
            else:
                _pad = pad
                trim = False

            _filter_shape = [filter_shape[1], filter_shape[0], filter_shape[2], filter_shape[3]]
            _W = W.transpose(1, 0, 2, 3)
            if isinstance(_pad, str): _pad = (_pad, _pad)
            dim_x, dim_y = tuple(get_input_shape(i, f, s, p, pa)
                                 for i, f, s, p, pa
                                 in zip([dim_x, dim_y], shape, stride,
                                    pool, _pad))

            input_shape = (batch_size, filter_shape[0], dim_x, dim_y)
            conv_out = T.nnet.abstract_conv.conv2d_grad_wrt_inputs(
                unpool_out, _W,
                filter_shape=_filter_shape,
                input_shape=input_shape,
                border_mode=_pad, subsample=stride, filter_flip=True)
            
            if trim:
                sx, sy = self.trims[l]
                conv_out = conv_out[:, :, sx:-sx, sy:-sy]

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

        outs['Xo'] = X
        X = X.reshape((X.shape[0], filter_shape[0] * dim_x * dim_y))
        outs['Xr'] = X
        if reshape is not None:
            X = X.reshape((reshape[0], reshape[1], X.shape[1]))
        outs['output'] = X

        return outs
    
    def dropout_epsilons(self, size, input_shape):
        epsilons = []
        
        for filter_shape, pool, pad, stride, n_filter in zip(
            self.filter_shapes, self.pools, self.pads, self.strides, self.n_filters):
            
            if isinstance(pad, str): pad = (pad, pad)
            dim_x, dim_y = tuple(get_input_shape(i, f, s, p, pa)
                                 for i, f, s, p, pa
                                 in zip(input_shape[1:], filter_shape, stride, pool,
                                        pad))
            
            input_shape = (dim_x, dim_y)
            shape = size + (n_filter, dim_x, dim_y)
            eps = self.trng.binomial(shape, p=1-self.dropout, n=1, dtype=floatX)
            epsilons.append(eps)

        return epsilons
    

_classes = {'CNN2D': CNN2D, 'RCNN2D': RCNN2D}
