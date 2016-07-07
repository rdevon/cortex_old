'''
Base Layer class.
'''

from collections import OrderedDict
import copy
import logging
import numpy as np
from pprint import pprint
import random
import re
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cortex.utils import floatX, _rng
from cortex.utils.tools import warn_kwargs, _p


logger = logging.getLogger(__name__)

def resolve_class(layer_type):
    try:
        C = _classes[layer_type]
    except KeyError:
        raise KeyError('Unexpected layer subclass `%s`, '
                       'available classes: %s' % (layer_type, _classes.keys()))
    return C

def build_layer(self, layer_type=None, **kwargs):
        factory = _classes[layer_type].factory
        return factory(**kwargs)

def init_weights(model, weight_noise=False, weight_scale=0.001, dropout=False,
                 **kwargs):
    '''Inialization function for weights.

    Args:
        model (Layer).
        weight_noise (bool): noise the weights.
        weight_scale (float): scale for weight initialization.
        dropout (bool): use dropout.
        **kwargs: extra kwargs.

    Returns:
        dict: extra kwargs.

    '''
    model.weight_noise = weight_noise
    model.weight_scale = weight_scale
    model.dropout = dropout
    return kwargs

def init_rngs(model, rng=None, trng=None, **kwargs):
    '''Initialization function for RNGs.

    Args:
        model (Layer).
        rng (np.randomStreams).
        trng (theano.randomStreams).
        **kwargs: extra kwargs.

    Returns:
        dict: extra kwargs.

    '''
    if rng is None:
        rng = _rng
    model.rng = rng
    if trng is None:
        model.trng = RandomStreams(random.randint(1, 10000))
    else:
        model.trng = trng
    return kwargs

def ortho_weight(ndim, rng=None):
    '''Make ortho weight tensor.

    '''
    if not rng:
        rng = _rng
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    '''Make normal weight tensor.

    '''
    if not rng:
        rng = _rng
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype(floatX)


class NoiseSwitch(object):
    '''Object to control noise of model.

    '''
    def __init__(self):
        self.noise = True

    def switch(self, switch_to):
        ons = ['on', 'On', 'ON']
        offs = ['off', 'Off', 'OFF']
        if switch_to in ons:
            self.noise = True
        elif switch_to in offs:
            self.noise = False

    def __call__(self):
        return self.noise


class Layer(object):
    '''Basic layer class.

    Attributes:
        name (str): name of layer.
        params (dict): dictionary of numpy.arrays
        excludes (list): list of parameters to exclude from learning.
        learn (bool): if False, do not change params.
        n_params (int): number of parameters

    '''
    _components = []
    _arg_map = {}
    _help = {}

    def __init__(self, name='', excludes=[], learn=True, noise_switch=None,
                 **kwargs):
        '''Init function for Layer.

        Args:
            name (str): name of layer.
            excludes (list): list of parameters to exclude from learning.
            learn (bool): if False, do not change params.
            noise_switch (NoiseSwitch).
            **kwargs: extra kwargs

        '''

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))

        self.logger.debug('Forming layer %r with name %s' % (
            self.__class__, name))

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        self.name = name
        self.params = None
        self.excludes = excludes
        self.learn = learn
        self.set_params()
        self.n_params = len(self.params)
        if noise_switch is None:
            noise_switch = NoiseSwitch()
        self.noise_switch = noise_switch

        warn_kwargs(self, kwargs)

        components = self.get_components()
        for component in components:
            component.noise_switch = self.noise_switch

        self.module = None

    def get_components(self):
        '''Gets layer components.

        '''
        components = []
        for k in self._components:
            component = getattr(self, k)
            if isinstance(component, list):
                components += [c for c in component if c is not None]
            elif component is not None:
                components.append(component)

        c_components = []
        for component in components:
            c_components += component.get_components()
        components += c_components
        return components

    def copy(self):
        '''Copy the Layer.

        '''
        return copy.deepcopy(self)

    def set_module(self, module):
        '''Sets layer module.

        '''
        self.module = module

    def set_params(self):
        '''Initialize the parameters.

        '''
        self.params = dict()

    def get_decay_params():
        '''Return parameters used in L1 and L2 decay.

        Returns:
            OrderedDict: dictionary of parameters.

        '''
        raise NotImplementedError('This layer (%s) does not implement decay'
                                  % self.__class__)

    def set_tparams(self):
        '''Sets the tensor parameters.

        '''
        if self.params is None:
            raise ValueError('Params not set yet')
        tparams = OrderedDict()

        for kk, pp in self.params.iteritems():
            tp = theano.shared(self.params[kk], name=kk)
            tparams[_p(self.name, kk)] = tp
            self.__dict__[kk] = tp

        return OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in [_p(self.name, e) for e in self.excludes])

    def get_excludes(self):
        '''Fetches the excluded parameters.

        '''
        if self.learn:
            return [_p(self.name, e) for e in self.excludes]
        else:
            return [_p(self.name, k) for k in self.params.keys()]

    def l1_decay(self, rate, **kwargs):
        '''L1 decay.

        Args:
            rate (float): decay rate.
            kwargs: keyword arguments of parameter name and rate.

        Returns:
            dict: dictionary of l1 decay costs for each parameter.

        '''
        decay_params = self.get_decay_params()

        cost = T.constant(0.).astype(floatX)
        rval = OrderedDict()
        if rate <= 0:
            return rval

        for k, v in decay_params.iteritems():
            if k in kwargs.keys():
                r = kwargs[k]
            else:
                r = rate
            self.logger.debug('Adding %.4g L1 decay to parameter %s' % (r, k))
            p_cost = r * (abs(v)).sum()
            rval[k + '_l1_cost'] = p_cost
            cost += p_cost

        rval = OrderedDict(
            cost = cost
        )

        return rval

    def l2_decay(self, rate, **kwargs):
        '''L2 decay.

        Args:
            rate (float): decay rate.
            kwargs: keyword arguments of parameter name and rate.

        Returns:
            dict: dictionary of l2 decay costs for each parameter.

        '''
        decay_params = self.get_decay_params()

        cost = T.constant(0.).astype(floatX)
        rval = OrderedDict()
        if rate <= 0:
            return rval
        for k, v in decay_params.iteritems():
            if k in kwargs.keys():
                r = kwargs[k]
            else:
                r = rate
            self.logger.debug('Adding %.4g L2 decay to parameter %s' % (r, k))
            p_cost = r * (v ** 2).sum()
            rval[k + '_l2_cost'] = p_cost
            cost += p_cost

        rval = OrderedDict(
            cost = cost
        )

        return rval

    def help(self):
        pprint(self._help)

    @classmethod
    def get_arg_reference(C, key, kwargs):
        try:
            k = C._arg_map[key]
        except KeyError:
            raise KeyError('Layer %s has no argument %s. Available arguments: '
                           '%s' % (C, key, C._arg_map))
        return kwargs[k]

    def __str__(self):
        attributes = self.__dict__
        attributes['params'] = dict(
            (k, '<numpy.ndarray: {shape: %s}>' % (a.shape,))
            for k, a in attributes['params'].iteritems())
        for k in ['trng', 'rng', 'logger']:
            attributes.pop(k, None)

        attr_str = ''
        for k, a in attributes.iteritems():
            if k in self._components:
                c_str = a.__str__()
                new_str = '\n\t'
                for i in range(0, len(c_str) - 1):
                    new_str += c_str[i]
                    if c_str[i] != '\t' and c_str[i + 1] == '\t':
                        new_str += '\t'
                new_str += c_str[-1]
            else:
                new_str = '\n\t%s: %s' % (k, a)
            attr_str += new_str

        s = ('<Layer %s: %s>' % (self.__class__.__name__, attr_str))
        return s

_classes = {'Layer': Layer}
from . import mlp, rnn, distributions
_modules = [mlp, rnn, distributions]
for module in _modules: _classes.update(**module._classes)