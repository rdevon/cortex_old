'''
Module for testing infrastructure
'''


from collections import OrderedDict
import numpy
import pprint
import theano
from theano import function
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from gru import CondGenGRU
from layers import BaselineWithInput
from layers import FFN
from mnist import mnist_iterator
from rbm import RBM
from trainer import get_grad
from trainer import train


floatX = theano.config.floatX

def test_main_model():
    import model as experiment
    model = experiment.get_model()
    data = model.pop('data')
    costs = experiment.get_costs(**model)
    f_grad_shared, f_update, cost_keys = get_grad('sgd', costs, **model)

    train = mnist_iterator(batch_size=2, mode='train')
    (x0, xT), _ = train.next()
    x0 = x0.reshape(1, train.dim)
    xT = xT.reshape(1, train.dim)
    inps = [x0, xT]
    rval = f_grad_shared(*inps)
    print rval
    assert False

def test_simple():
    dim_r = 19
    dim_g = 13
    batch_size = 3
    n_steps = 7

    train = mnist_iterator(batch_size=2 * batch_size, mode='train')

    X0 = T.matrix('x0', dtype=floatX)
    XT = T.matrix('x0', dtype=floatX)

    trng = RandomStreams(6 * 10 * 2015)

    dim_in = train.dim

    rnn = CondGenGRU(dim_in, dim_r, trng=trng, stochastic=False)
    rbm = RBM(dim_in, dim_g, trng=trng, stochastic=False)

    tparams = rnn.set_tparams()
    tparams.update(rbm.set_tparams())

    outs = OrderedDict()
    outs_rnn, updates = rnn(X0, XT, reversed=True, n_steps=n_steps)
    outs[rnn.name] = outs_rnn

    outs_rbm, updates_rbm = rbm.energy(outs[rnn.name]['x'])
    outs[rbm.name] = outs_rbm
    updates.update(updates_rbm)

    xs, _ = train.next()
    x0 = xs[:batch_size]
    xT = xs[batch_size:]

    inps = [x0, xT]

    fn = theano.function([X0, XT], outs[rnn.name]['x'].shape)
    print fn(*inps)

    print 'here'

    fn = theano.function([X0, XT], outs[rbm.name]['log_p'])
    print fn(*inps)

def test_baseline():
    X0 = T.matrix('x0', dtype=floatX)
    XT = T.matrix('xT', dtype=floatX)

    train = mnist_iterator(batch_size=20, mode='train')
    x, _ = train.next()
    x0 = x[:10]
    xT = x[10:]

    a = theano.shared(x0)
    b = theano.shared(xT)
    ab = T.concatenate([a[None, :, :], b[None, :, :]])


    def step(x, y):
        return x + y

    out, update = theano.scan(step, sequences=[ab, ab], strict=True)
    print out.shape.eval()

    inps = [x0, xT]
    baseline = BaselineWithInput((train.dim, train.dim))
    baseline.set_tparams()

    ffn = FFN(train.dim, 10)
    ffn.set_tparams()
    outs, updates = ffn(X0)

    z = outs['z']
    outs_bl, updates_bl = baseline(z, X0, XT)
    updates.update(updates_bl)

    fn = theano.function([X0, XT], outs_bl['x_centered'], updates=updates)
    print fn(x0, xT)