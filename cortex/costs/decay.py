
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