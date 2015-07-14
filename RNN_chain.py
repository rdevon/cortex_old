'''
Sampling and inference with LSTM models
'''

from collections import OrderedDict
import matplotlib
from matplotlib import animation
from matplotlib import pylab as plt
import numpy as np
import os
from sys import stdout
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from gru import GenGRU
from rnn import GenRNN
from layers import BaselineWithInput
from mnist import mnist_iterator
import op
from tools import itemlist

floatX = theano.config.floatX


def visualize(model_file, mode='gru'):
    train = mnist_iterator(batch_size=1, mode='train', inf=False)

    trng = RandomStreams(6 * 23 * 2015)
    dim_in = train.dim
    dim_h=500
    n_inference_steps=30

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError()

    rnn = C(dim_in, dim_h, trng=trng, h0_mode='ffn')

    pretrained_model = np.load(model_file)

    for k, v in rnn.params.iteritems():
        try:
            pretrained_v = pretrained_model[
                '{name}_{key}'.format(name=rnn.name, key=k)]
            rnn.params[k] = pretrained_v
        except KeyError:
            print '{} not found'.format(k)

    tparams = rnn.set_tparams()

    X = T.matrix('x', dtype=floatX)
    p0 = T.zeros_like(X) + X
    h0 = T.dot(X, rnn.W0) + rnn.b0

    x_s, p_s, h_s = rnn.step_sample(X, p0, h0, *rnn.get_non_seqs())

    f_h0 = theano.function([X], [x_s, p_s])

    ps = []
    try:
        x, _ = train.next()
        x = x
        s = 0
        while True:
            stdout.write('\rSampling (%d): Press ^c to stop' % s)
            stdout.flush()
            x, p = f_h0(x)
            ps.append(p)
            s += 1
    except KeyboardInterrupt:
        print 'Exiting'

    fig = plt.figure()
    data = np.zeros((28, 28))
    im = plt.imshow(data, vmin=0, vmax=1)

    def init():
        im.set_data(np.zeros((28, 28)))

    def animate(i):
        data = ps[i].reshape((28, 28))
        im.set_data(data)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=s,
                                   interval=1)
    plt.show()

    return

def binary_energy(x0, x1, model, mode='rnn'):
    if model.h0_mode == 'average':
        h0 = T.alloc(0., model.dim_h).astype('float32') + model.h0
    elif model.h0_mode == 'ffn':
        h0 = T.dot(x0, model.W0) + model.b0
    elif model.h0_mode is None:
        h0 = T.alloc(0., model.dim_h)
    else:
        raise ValueError(model.h0_mode)

    if mode == 'rnn':
        h1 = T.tanh(T.dot(h0, model.Ur) + T.dot(x0, model.XH) + model.bh)
        p = T.nnet.sigmoid(T.dot(h1, model.HX) + model.bx)
    elif mode == 'gru':
        return (x0 + x1) * 0

    energy = (-x1 * T.log(p + 1e-7) - (1 - x1) * T.log(1 - p + 1e-7)).sum()
    return energy

def energy_function(model, mode='rnn'):
    x0 = T.vector('x0', dtype=floatX)
    x1 = T.vector('x1', dtype=floatX)
    energy = binary_energy(x0, x1, model, mode=mode)

    return theano.function([x0, x1], energy)

def test(batch_size=20, dim_h=500, save_graphs=False, mode='rnn'):
    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True,
                           out_mode='model_chains')

    dim_in = train.dim
    X = T.tensor3('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError()

    rnn = C(dim_in, dim_h, trng=trng, h0_mode='ffn')
    tparams = rnn.set_tparams()
    train.set_f_energy(energy_function(rnn, mode=mode))

    mask = T.alloc(1., 2).astype('float32')

    X_s = T.zeros_like(X)
    X_s = T.set_subtensor(X_s[1:], X[:-1])
    outs, updates = rnn(X_s)
    h = outs['h']
    p = outs['p']
    x = outs['x']

    consider_constant = []

    energy = -(X * T.log(p + 1e-7) + (1 - X) * T.log(1 - p + 1e-7)).sum(axis=(0, 2))
    cost = energy.mean()

    consider_constant = [x]

    if rnn.h0_mode == 'ffn':
        print 'Using a ffn h0 with input x0'
        h0 = T.dot(X.reshape((X.shape[0] * X.shape[1], X.shape[2])), rnn.W0) + rnn.b0
        ht = h.reshape((h.shape[0] * h.shape[1], h.shape[2]))
        ht_c = T.zeros_like(ht) + ht
        h0_cost = ((ht_c - h0)**2).sum(axis=1).mean()
        cost += h0_cost
        consider_constant.append(ht_c)

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    out_s, updates_s = rnn.sample(X[0, :batch_size], n_samples=batch_size, n_steps=40)
    updates.update(updates_s)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    optimizer = 'adam'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=[energy.mean(), outs['x'], out_s['p'], h.mean(), abs(h).mean(), h.std(axis=0).mean(), (h[0] - h[-1]).mean()])

    print 'Actually running'
    learning_rate = 0.00001

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x)
            r = False
            for k, out in zip(['cost', 'energy'], rval):
                if np.any(np.isnan(out)):
                    print k, 'nan'
                    r = True
                elif np.any(np.isinf(out)):
                    print k, 'inf'
                    r = True
            if r:
                return
            if e % 10 == 0:
                print ('%d: cost: %.5f | energy: %.5f | prob: %.5f | '
                       'h_mean: %.5f | h_abs_mean: %.5f | h_std_mean: %.5f | '
                       'h_diff: %.5f'
                       % (e, rval[0], rval[1], np.exp(-rval[1]), rval[4], rval[5], rval[6], rval[7]))
            if e % 10 == 0:
                idx = np.random.randint(rval[2].shape[1])
                sample = np.concatenate([x[:, idx, :][None, :, :],
                                        rval[2][:, idx, :][None, :, :]], axis=0)
                train.save_images(sample, '/Users/devon/tmp/chain_sampler2.png')
                sample_chain = rval[3]
                train.save_images(sample_chain, '/Users/devon/tmp/chain_chain2.png')
                train.save_images(x, '/Users/devon/tmp/input_chain2.png')
                #prob_chain = rval[3]
                #train.save_images(prob_chain, '/Users/devon/tmp/grad_probs.png')

            f_grad_updates(learning_rate)
    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = os.path.join('/Users/devon/tmp/',
                           'rnn_chain_model_{}.npz'.format(int(time.time())))

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

if __name__ == '__main__':
    test()