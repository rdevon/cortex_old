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

from gru import CondGenGRU
from gru import GRU
from gru import HeirarchalGRU
from gru import SimpleInferGRU
from layers import BaselineWithInput
from layers import FFN
from layers import Logistic
from mnist import mnist_iterator
from rbm import RBM
from tools import itemlist
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
    n_reps = 5

    train = mnist_iterator(batch_size=2 * batch_size, mode='train', repeat=n_reps)

    X = T.matrix('x', dtype='float32')
    X0 = X[:batch_size * n_reps]
    XT = X[batch_size * n_reps:][::-1]

    xs, _ = train.next()

    trng = RandomStreams(6 * 10 * 2015)

    dim_in = train.dim

    rnn = CondGenGRU(dim_in, dim_r, trng=trng)
    rbm = RBM(dim_in, dim_g, trng=trng, param_file='rbm_model.yaml', learn=False)
    baseline = layers.BaselineWithInput((dim_in, dim_in), n_steps,
        name='reward_baseline')

    tparams = rnn.set_tparams()
    tparams.update(rbm.set_tparams())
    tparams.update(baseline.set_tparams())

    outs = OrderedDict()
    outs_rnn, updates = rnn(X0, XT, reverse=True, n_steps=n_steps)
    outs[rnn.name] = outs_rnn

    outs_rbm, updates_rbm = rbm.energy(outs[rnn.name]['x'])
    outs[rbm.name] = outs_rbm
    updates.update(updates_rbm)

    q = outs[rnn.name]['p']
    x = outs[rnn.name]['x']

    fn = theano.function([X], x)
    s = fn(xs)
    train.save_images(s, path.join('/Users/devon/tmp/', 'test_samples.png'))

    outs_rnn_e, updates_rnn_e = rnn.energy(x, q)
    outs[rnn.name].update(outs_rnn_e)
    updates.update(updates_rnn_e)

    acc_log_q = outs[rnn.name]['acc_log_p']
    acc_log_p = outs[rbm.name]['acc_log_p']
    reward = (acc_log_p - acc_log_q)

    outs_baseline, updates_baseline = baseline(reward, True, X0, XT)
    outs[baseline.name] = outs_baseline
    updates.update(updates_baseline)

    inps = [xs]

    fn = theano.function([X], reward.shape)
    print fn(*inps)

    fn = theano.function([X], outs[baseline.name]['x_centered'])

    print fn(*inps)
    idb = outs[baseline.name]['idb']
    m = outs[baseline.name]['m']
    var = outs[baseline.name]['var']
    reward0 = outs[baseline.name]['x']
    idb_cost = (((reward0 - idb - m) / T.maximum(1., T.sqrt(var)))**2).mean()

    fn = theano.function([X], idb_cost)
    print fn(*inps)

    centered_reward = outs[baseline.name]['x_centered']
    fn = theano.function([X], centered_reward.shape)
    print fn(*inps)

    log_q = outs['cond_gen_gru']['log_p']
    log_p = outs['rbm']['log_p']

    base_cost = -(log_p + centered_reward * log_q).mean()
    fn = theano.function([X], base_cost, updates=updates)
    print fn(*inps)
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
    mask_n = outs['mask_n']

    fn = theano.function([X], [r_hat, mask_n])
    r_hat, mask = fn(x)
    print mask
    print r_hat
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
    fn = theano.function([], [outs['x'], outs['h'], outs['p'], outs['q']], updates=updates)
    print fn()

    assert False

def test_inference(batch_size=1, dim_h=10, l=.1):
    import op
    train = mnist_iterator(batch_size=2*batch_size, mode='train')
    dim_in = train.dim

    X = T.tensor3('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)
    rnn = SimpleInferGRU(dim_in, dim_h, trng=trng)
    tparams = rnn.set_tparams()
    mask = T.alloc(1., 2).astype('float32')
    #(x_hats, energies), updates = rnn.inference(X, mask, l,
    #                                            n_inference_steps=1000)
    (x_hats, energies), updates = rnn.inference(X, mask, l, n_inference_steps=10)
    grads = T.grad(energies[-1], wrt=itemlist(tparams))

    lr = T.scalar(name='lr')
    optimizer = 'rmsprop'

    #chain, updates_s = rnn.sample(X)
    #updates.update(updates_s)

    x, _ = train.next()
    x = x.reshape(2, batch_size, x.shape[1]).astype(floatX)
    fn = theano.function([X], [x_hats] + grads, updates=updates)
    print fn(x)
    assert False
    '''
    fn = theano.function([X], [XO, h], updates=updates)
    #theano.printing.debugprint(energies[0])
    xo, h = fn(x)
    print xo.shape
    print h.shape
    #print es
    #train.save_images(x0, '/Users/devon/tmp/grad_sampler.png')
    '''

    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], energies[-1],
        extra_ups=updates,
        extra_outs=[])

    print 'Actually running'
    learning_rate = 0.1
    for e in xrange(10):
        x, _ = train.next()
        x = x[:, None, :].astype(floatX)
        rval = f_grad_shared(x)
        r = False
        for k, out in zip(['energy'], rval):
            if np.any(np.isnan(out)):
                print k, 'nan'
                r = True
            elif np.any(np.isinf(out)):
                print k, 'inf'
                r = True
        if r:
            return
        if e % 10 == 0:
            print e, rval[0]

        f_grad_updates(learning_rate)

    assert False

def test_twitter_batches():
    from tweet_rnn_model import get_model
    model = get_model()
    train = model['data']['train']
    x, r, m = train.next()
    print x.shape, r.shape, m.shape

    mask = model['outs']['hiero_gru']['mask']
    r_hat = model['outs']['logistic']['y_hat'][:, :, 0]
    X = model['inps']['x']
    R = model['inps']['r']
    M = model['inps']['m']
    updates = model['updates']
    threshold = 0.3

    mask = 1 - mask[1:]
    r_hat = r_hat[1:]
    R_ = R[1:]

    n_total = T.floor(threshold * mask.sum(axis=0).astype('float32')).astype('int64')
    #fn = theano.function([X, M], [mask, r_hat, n_total])

    #print fn(x, m)
    #assert False
    GT_ids = T.argsort(R_ * mask, axis=0)
    RH_ids = T.argsort(r_hat * mask, axis=0)

    fn = theano.function([X, R, M], [mask.shape, RH_ids.shape, GT_ids.shape, r_hat.shape, R_.shape], updates=updates)
    print fn(x, r, m)
    #assert False

    def step(a, idx, n):
        b = T.zeros_like(a)
        idx = idx[-n:]
        return T.set_subtensor(b[idx], 1.)

    mask_gt, _ = theano.scan(
        step,
        sequences=[mask.T, GT_ids.T, n_total],
        outputs_info=[None],
        non_sequences=[],
        name='top_step',
        strict=True)

    mask_r, _ = theano.scan(
        step,
        sequences=[mask.T, RH_ids.T, n_total],
        outputs_info=[None],
        non_sequences=[],
        name='top_step',
        strict=True)

    mask_final = mask_gt * mask_r
    acc = (mask_final.sum(axis=1).astype('float32') / n_total.astype('float32')).mean()

    fn = theano.function([X, R, M], [mask_final.shape, acc], updates=updates)
    print fn(x, r, m)
    assert False

    return acc

def test_heir_wo():
    from twitter_rnn_ffn import get_model
    model = get_model()
    train = model['data']['train']
    x, r, m = train.next()

    mask = model['outs']['heiro_gru_wo']['mask']
    h = model['outs']['heiro_gru_wo']['h']
    mask_n = model['outs']['heiro_gru_wo']['mask_n']

    o = model['outs']['heiro_gru_wo']['o']

    #o_c = mask_n[1:].compress(o, axis=0)#.dimshuffle(1, 0, 2)
    X = model['inps']['x']
    R = model['inps']['r']
    M = model['inps']['m']
    updates = model['updates']

    fn = theano.function([X, M], [h.shape, o.shape])
    print fn(x, m)
    assert False

def test_tweet_gen():
    from tweet_gen_model import get_model
    model = get_model()
    train = model['data']['train']
    x, m = train.next()

    X = model['inps']['x']
    M = model['inps']['m']
    updates = model['updates']
    z = model['outs']['logit']['z']
    y = model['outs']['logit']['y']
    X_hat = model['outs']['softmax']['y_hat']

    fn = theano.function([X, M], [X_hat.shape, X.shape], updates=updates)

    print fn(x, m)
    assert False