'''
Base Layer class.
'''

from collections import OrderedDict
import copy
import logging
import theano
from theano import tensor as T

from cortex.utils import floatX
from cortex.utils.tools import (
    warn_kwargs,
    _p
)


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

    def __init__(self, name='', excludes=[], learn=True, **kwargs):
        '''Init function for Layer.

        Args:
            name (str): name of layer.
            excludes (list): list of parameters to exclude from learning.
            learn (bool): if False, do not change params.
            **kwargs: extra kwargs

        '''

        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))

        self.logger.debug('Forming layer %r with name %s' % (
            self.__class__, name))

        self.name = name
        self.params = None
        self.excludes = excludes
        self.learn = learn
        self.set_params()
        self.n_params = len(self.params)
        warn_kwargs(self, kwargs)

    def copy(self):
        '''Copy the Layer.

        '''
        return copy.deepcopy(self)

    def set_params(self):
        '''Initialize the parameters.

        '''
        self.params = dict()

    def get_decay_params():
        '''Return parameters used in L1 and L2 decay.

        Returns:
            OrderedDict: dictionary of parameters.

        '''
        return OrderedDict()

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
