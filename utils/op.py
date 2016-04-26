'''
Optimization routines.

Based (and copied) on Kyunghyun Cho's arctic repo.
'''
import ipdb
import numpy as np
import theano
from theano import tensor as T
import tools
from collections import OrderedDict


profile = False

def adam3(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
          exclude_params=set([])):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
               for k, p in tparams.iteritems()]
    '''
    g_norm = 0.

    for i in xrange(len(grads)):

        grads[i] /= T.cast(100, dtype=theano.config.floatX)
        g_norm += (grads[i]**2).sum()

    g_norm = T.sqrt(g_norm)
    scaler = 5 / T.maximum(5, g_norm)

    for i in xrange(len(grads)):
        grads[i] *= scaler
    '''
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=gsup+extra_ups, profile=profile)

    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lambd = (1 - 1e-8)

    updates = OrderedDict()

    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = b1**(i_t)
    fix2 = b2**(i_t)
    b1_t = b1 * lambd**i
    b2_t = b2 * lambd**i

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = b1_t * m + (1. - b1_t) * g
        v_t = b2_t * v + (1. - b2_t) * g**2

        m_t_hat = m_t / (1. - fix1)
        v_t_hat = v_t / (1. - fix2)

        g_t = m_t_hat / (T.sqrt(v_t_hat) + eps)

        p_t = p - lr * g_t
        updates[m] =  m_t
        updates[v] =  v_t
        if p.name not in exclude_params:
            updates[p] = p_t

    '''
    for k, updated_param in updates.items():
        if 'W' in str(k):
            col_norms = T.sqrt(T.sqr(updated_param).sum(axis=0))
            desired_norms = T.clip(col_norms, 0, 1.9365)
            ratio = (desired_norms / (1e-8 + col_norms))
            updates[k] = updated_param * ratio
    '''
    updates[i] = i_t

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(lr, [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adam2(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
          exclude_params=set([])):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
               for k, p in tparams.iteritems()]
    '''
    g_norm = 0.

    for i in xrange(len(grads)):

        grads[i] /= T.cast(100, dtype=theano.config.floatX)
        g_norm += (grads[i]**2).sum()

    g_norm = T.sqrt(g_norm)
    scaler = 5 / T.maximum(5, g_norm)

    for i in xrange(len(grads)):
        grads[i] *= scaler
    '''
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=gsup+extra_ups, profile=profile)

    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lambd = (1 - 1e-8)

    updates = OrderedDict()

    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = b1**(i_t)
    fix2 = b2**(i_t)
    b1_t = b1 * lambd**i

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = b1_t * m + (1. - b1_t) * g
        v_t = b2 * v + (1. - b2) * g**2

        m_t_hat = m_t / (1. - fix1)
        v_t_hat = v_t / (1. - fix2)

        g_t = m_t_hat / (T.sqrt(v_t_hat) + eps)

        p_t = p - lr * g_t
        updates[m] =  m_t
        updates[v] =  v_t
        if p.name not in exclude_params:
            updates[p] = p_t

    '''
    for k, updated_param in updates.items():
        if 'W' in str(k):
            col_norms = T.sqrt(T.sqr(updated_param).sum(axis=0))
            desired_norms = T.clip(col_norms, 0, 1.9365)
            ratio = (desired_norms / (1e-8 + col_norms))
            updates[k] = updated_param * ratio
    '''
    updates[i] = i_t

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(lr, [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adam(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
         exclude_params=set([])):
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
               for k, p in tparams.iteritems()]

    '''
    g_norm = 0.

    for i in xrange(len(grads)):

        grads[i] /= T.cast(100, dtype=theano.config.floatX)
        g_norm += (grads[i]**2).sum()

    g_norm = T.sqrt(g_norm)
    scaler = 5 / T.maximum(5, g_norm)

    for i in xrange(len(grads)):
        grads[i] *= scaler
    '''

    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=gsup+extra_ups, profile=profile)

    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    updates = OrderedDict()

    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = b1**(i_t)
    fix2 = b2**(i_t)
    lr_t = lr * T.sqrt(1-fix2) / (1-fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = b1 * m + (1. - b1) * g
        v_t = b2 * v + (1. - b2) * g**2
        g_t = lr_t * m_t / (T.sqrt(v_t) + eps)
        p_t = p - g_t
        updates[m] =  m_t
        updates[v] =  v_t
        if p.name not in exclude_params:
            updates[p] = p_t

    for k, updated_param in updates.items():
        if 'W' in str(k):
            col_norms = T.sqrt(T.sqr(updated_param).sum(axis=0))
            desired_norms = T.clip(col_norms, 0, 1.9365)
            ratio = (desired_norms / (1e-8 + col_norms))
            updates[k] = updated_param * ratio

    updates[i] = i_t

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(lr, [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
             exclude_params=set([])):
    '''Adadelta'''
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rup2'%k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2'%k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
        for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=zgup+rg2up+extra_ups, profile=profile)

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
        for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tools.itemlist(tparams), updir)
        if p.name not in exclude_params]

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(lr, [], updates=ru2up+param_up,
        on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def rmsprop(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
            exclude_params=set([]),
            relaxation=1e-4, momentum=0.9, coefficient=0.95
            ):
    '''RMSProp'''
    print ('RMSprop with relaxation %.5f, momentum %.2f, and coeffient %.2f'
           % (relaxation, momentum, coefficient))
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad'%k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2'%k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, coefficient * rg + (1.0 - coefficient) * g)
        for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, coefficient * rg2 + (1.0 - coefficient) * (g ** 2))
        for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=zgup+rgup+rg2up+extra_ups, profile=profile)

    updir = [theano.shared(p.get_value() * np.float32(0.), name='%s_updir'%k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, momentum * ud - lr * zg / T.sqrt(rg2 - rg ** 2 + relaxation))
        for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    param_up = [(p, p + udn[1]) for p, udn in zip(tools.itemlist(tparams), updir_new)
        if p.name not in exclude_params]

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(
        lr, [], updates=updir_new+param_up, on_unused_input='ignore',
        profile=profile)

    return f_grad_shared, f_update

def sgd(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
        exclude_params=set([])):
    '''Stochastic gradient descent'''
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k)
               for k, p in tparams.iteritems()]

    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=gsup+extra_ups, profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(tools.itemlist(tparams), gshared)
        if p.name not in exclude_params]

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(lr, [], updates=pup, profile=profile)

    return f_grad_shared, f_update

def rmsprop2(lr, tparams, grads, inp, cost, extra_ups=[], extra_outs=[],
             exclude_params=set([]),
            relaxation=1e-4, momentum=0.9, coefficient=0.95):
    '''An alternative RMSProp'''
    print 'RMSprop with relaxation %.5f, momentum %.2f, and coeffient %.2f' % (relaxation, momentum, coefficient)
    zipped_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_grad'%k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad'%k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * np.float32(0.), name='%s_rgrad2'%k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, coefficient * rg + (1.0 - coefficient) * g)
        for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, coefficient * rg2 + (1.0 - coefficient) * (g ** 2))
        for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(
        inp, [cost]+extra_outs, updates=zgup+rgup+rg2up+extra_ups, profile=profile)

    updir = [theano.shared(p.get_value() * np.float32(0.), name='%s_updir'%k)
             for k, p in tparams.iteritems()]
    updir_temp = [momentum * ud - lr * zg / T.sqrt(rg2 - rg ** 2 + relaxation)
                  for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                             running_grads2)]

    for i, (k, updated_param) in enumerate(zip(updir, updir_temp)):
        if 'W' in str(k):
            col_norms = T.sqrt(T.sqr(updated_param).sum(axis=0))
            desired_norms = T.clip(col_norms, 0, 1.9365)
            ratio = (desired_norms / (1e-8 + col_norms))
            updir_temp[i] = updated_param * ratio

    #updir_new = [(ud, momentum * ud - lr * zg / T.sqrt(rg2 - rg ** 2 + relaxation)) for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads, running_grads2)]
    updir_new = [(ud, ud_new) for ud, ud_new in zip(updir, updir_temp)]
    param_up = [(p, p + udn[1]) for p, udn in zip(tools.itemlist(tparams), updir_new)
        if p.name not in exclude_params]

    if not isinstance(lr, list): lr = [lr]
    f_update = theano.function(
        lr, [], updates=updir_new+param_up, on_unused_input='ignore',
        profile=profile)

    return f_grad_shared, f_update
