'''
Module for testing infrastructure
'''


from collections import OrderedDict
import numpy as np
from os import path
import pprint
import theano
from theano import function
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from gru import GRU
from gru import CondGenGRU
from gru import HeirarchalGRU
from layers import BaselineWithInput
from layers import FFN
from layers import Logistic
from mnist import mnist_iterator
from rbm import RBM
from trainer import get_grad
from trainer import train
from twitter_api import TwitterFeed


floatX = theano.config.floatX

def test_main_model():
    import model as experiment

    train = mnist_iterator(batch_size=2, mode='train')
    (x0, xT), _ = train.next()
    x0 = x0.reshape(1, train.dim)
    xT = xT.reshape(1, train.dim)
    inps = [x0, xT]

    model = experiment.get_model()
    data = model.pop('data')
    costs = experiment.get_costs(**model)

    f_grad_shared, f_update, cost_keys = get_grad('sgd', costs, **model)

    rval = f_grad_shared(*inps)

    assert False

def test_simple():
    dim_r = 19
    dim_g = 13
    batch_size = 3
    n_steps = 7

    train = mnist_iterator(batch_size=2 * batch_size, mode='train')

    X0 = T.matrix('x0', dtype=floatX)
    XT = T.matrix('x0', dtype=floatX)

    xs, _ = train.next()
    x0 = xs[:batch_size]
    xT = xs[batch_size:]

    trng = RandomStreams(6 * 10 * 2015)

    dim_in = train.dim

    rnn = CondGenGRU(dim_in, dim_r, trng=trng, stochastic=False)
    rbm = RBM(dim_in, dim_g, trng=trng, stochastic=False)
    baseline = BaselineWithInput((train.dim, train.dim), n_steps + 1)

    tparams = rnn.set_tparams()
    tparams.update(rbm.set_tparams())
    tparams.update(baseline.set_tparams())

    outs = OrderedDict()
    outs_rnn, updates = rnn(X0, XT, reversed=True, n_steps=n_steps)
    outs[rnn.name] = outs_rnn

    outs_rbm, updates_rbm = rbm.energy(outs[rnn.name]['x'])
    outs[rbm.name] = outs_rbm
    updates.update(updates_rbm)

    q = outs[rnn.name]['p']
    samples = outs[rnn.name]['x']

    fn = theano.function([X0, XT], samples)
    s = fn(x0, xT)

    train.save_images(s, path.join('/Users/devon/tmp/', 'test_samples.png'))
    energy_q = (samples * T.log(q + 1e-7) + (1. - samples) * T.log(1. - q + 1e-7)).sum(axis=(0, 2))
    outs[rnn.name]['log_p'] = energy_q
    energy_p = outs[rbm.name]['log_p']
    reward = (energy_p - energy_q)

    outs_baseline, updates_baseline = baseline(reward, X0, XT)
    outs[baseline.name] = outs_baseline
    updates.update(updates_baseline)

    inps = [x0, xT]

    fn = theano.function([X0, XT], reward.shape)
    print fn(*inps)

    fn = theano.function([X0, XT], outs[baseline.name]['x_centered'])

    print fn(*inps)
    idb = outs[baseline.name]['idb']
    c = outs[baseline.name]['c']
    idb_cost = ((reward - idb - c)**2).mean()

    fn = theano.function([X0, XT], idb_cost)
    print fn(x0, xT)

    centered_reward = outs[baseline.name]['x_centered']
    fn = theano.function([X0, XT], centered_reward.shape)
    print fn(x0, xT)

    base_cost = -(energy_p + centered_reward * energy_q).mean()
    fn = theano.function([X0, XT], base_cost)
    print fn(x0, xT)
    assert False

def test_baseline():
    X0 = T.matrix('x0', dtype=floatX)
    XT = T.matrix('xT', dtype=floatX)

    train = mnist_iterator(batch_size=26, mode='train')
    x, _ = train.next()
    x0 = x[:13]
    xT = x[13:]

    inps = [x0, xT]

    baseline = BaselineWithInput((train.dim, train.dim))
    baseline.set_tparams()

    A = X0.dot(baseline.w0) + XT.dot(baseline.w1)

    fn = theano.function([X0, XT], A)
    a = fn(x0, xT)
    print a, a.shape

    ffn = FFN(train.dim, 11)
    ffn.set_tparams()
    outs, updates = ffn(X0)

    z = outs['z']
    outs_bl, updates_bl = baseline(z, X0, XT)
    updates.update(updates_bl)

    fn = theano.function([X0, XT], outs_bl['x_centered'], updates=updates)
    print fn(x0, xT)

def test_mask(batch_size=11):
    train = mnist_iterator(batch_size=batch_size, mode='train')
    x, _ = train.next()
    x = np.concatenate([x, np.zeros_like(x)]).astype('float32')
    x = x.reshape((x.shape[0], 1, x.shape[1]))
    mask = np.zeros((x.shape[0], 1)).astype('float32')
    print mask.shape, x.shape
    mask[:batch_size] = 1.

    X = T.tensor3('X')
    M = T.matrix('M')

    rnn = GRU(train.dim, 7)
    rnn.set_tparams()
    outs, updates = rnn(X, M)

    fn = theano.function([X, M], outs['h'])
    print fn(x, mask)

def test_heirarchal_gru(batch_size=1, dim_h=11, dim_s=7):
    train = TwitterFeed()
    x, r = train.next()
    X = T.tensor3('X')
    rnn = HeirarchalGRU(train.dim, dim_h, dim_s)
    rnn.set_tparams()
    outs, updates = rnn(X)

    #fn = theano.function([X], [outs['h'], outs['hs'], outs['o']])
    #out = fn(x)
    #a = np.array(np.where(x[:, 0, :] == 1.)[1].tolist()[0])[:30]
    #print zip(a, out[0][:30, 0, 0], out[1][:30, 0, 0])

    logistic = Logistic()
    outs_l, _ = logistic(outs['o'])
    r_hat = outs_l['y_hat']
    mask = outs['mask']

    fn = theano.function([X], [r_hat, mask])
    r_hat, mask = fn(x)
    print ((r_hat[:, :, 0] - r) * (1 - mask)).sum() / r_hat.shape[0].asfloat()

    #print x[:, :, 0].shape
    #print a
    #print out[2][:30, 0]

    assert False

def test_rbm(batch_size=7, n_steps=11, dim_in=17, dim_h=13):
    trng = RandomStreams(6 * 10 * 2015)
    rbm = RBM(dim_in, dim_h, trng=trng)
    rbm.set_tparams()

    outs, updates = rbm(n_steps, n_chains=batch_size)
    fn = theano.function([], outs['x'], updates=updates)
    print fn()

    assert False