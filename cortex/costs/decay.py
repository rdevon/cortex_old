'''Weight decay costs

'''

from collections import OrderedDict
import logging
from theano import tensor as T

from ..utils import floatX


logger = logging.getLogger(__name__)

def l1_decay(rate, *tparams):
    '''L1 decay.

    Args:
        rate (float): decay rate.
        kwargs: keyword arguments of parameter name and rate.

    Returns:
        dict: dictionary of l1 decay costs for each parameter.

    '''
    cost = T.constant(0.).astype(floatX)
    rval = OrderedDict()
    if rate <= 0:
        return rval

    for tparam in tparams:
        logger.debug('Adding %.4g L1 decay to parameter %s'
                     % (rate, tparam.name))
        p_cost = rate * (abs(tparam)).sum()
        rval[tparam.name + '_l1_cost'] = p_cost
        cost += p_cost

    rval['cost'] = cost

    return rval

def l2_decay(rate, *tparams):
    '''L2 decay.

    Args:
        rate (float): decay rate.
        kwargs: keyword arguments of parameter name and rate.

    Returns:
        dict: dictionary of l2 decay costs for each parameter.

    '''
    cost = T.constant(0.).astype(floatX)
    rval = OrderedDict()
    if rate <= 0:
        return rval

    for tparam in tparams:
        logger.debug('Adding %.4g L2 decay to parameter %s'
                     % (rate, tparam.name))
        p_cost = rate * (tparam ** 2).sum()
        rval[tparam.name + '_l2_cost'] = p_cost
        cost += p_cost

    rval['cost'] = cost

    return rval

_costs = {'l1_decay': l1_decay, 'l2_decay': l2_decay}