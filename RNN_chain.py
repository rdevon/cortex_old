'''
Sampling and inference with LSTM models
'''

import argparse
from collections import OrderedDict
from glob import glob
import matplotlib
from matplotlib import animation
from matplotlib import pylab as plt
import numpy as np
import os
from os import path
import pprint
import random
import shutil
import sys
from sys import stdout
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import yaml

from GSN import likelihood_estimation_parzen as lep
from gru import GenGRU
from rbm import RBM
from rnn import GenRNN
from horses import Horses
from horses import SimpleHorses
from layers import Averager
from layers import BaselineWithInput
from layers import MLP
from layers import ParzenEstimator
from mnist import mnist_iterator
from mnist import MNIST_Chains
import op
import tools
from tools import check_bad_nums
from tools import itemlist
from tools import load_model
from tools import log_mean_exp
from tools import parzen_estimation


floatX = theano.config.floatX


def estimate_parzen(samples, epochs=1000, optimizer='adam',
                    learning_rate=0.001):

    test = mnist_iterator(batch_size=5000, mode='test', inf=False)
    valid = mnist_iterator(batch_size=5000, mode='valid', inf=False)

    parzen = ParzenEstimator(samples.shape[-1])

    x_v, _ = valid.next()
    print 'Setting sigmas'
    parzen.set_params(samples, x_v)

    tparams = parzen.set_tparams()

    S = T.matrix('s', dtype=floatX)
    X = T.matrix('x', dtype=floatX)
    L = parzen(S, X)

    f_par = theano.function([S, X], L)
    x_t, _ = test.next()

    print 'Getting log likeihood estimate'
    log_est = f_par(samples, x_t)
    print 'Parzen log likelihood lower bound: %.5f' % log_est

def lower_bound(model_file, n_samples=10000, sigma=0.2, from_chain=False,
                start_at_data=False):

    models, kwargs = load_model(model_file, unpack)
    dataset_args = kwargs['dataset_args']
    dataset = kwargs['dataset']

    rnn, train, test, f_h0 = load_model_for_sampling(model_file)
    params = rnn.get_sample_params()

    if dataset == 'mnist':
        test = MNIST_Chains(batch_size=n_samples, mode='test', **dataset_args)
        valid = MNIST_Chains(batch_size=n_samples / 10, mode='valid', **dataset_args)
        test.randomize()
        valid.randomize()
    elif dataset == 'horses':
        test = Horses(batch_size=1, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)

    h_s, x_s, p_s = rnn.step_sample(H, X, *params)
    f_sam = theano.function([X, H], [x_s, h_s, p_s])

    if from_chain:
        xs = []
        x = train.next_simple(batch_size=1)
        h = f_h0(x)
        while len(xs) < n_samples:
            x, h, p = f_sam(x, h)
            xs.append(p)
        samples = np.array(xs)[:, 0]
    else:
        print 'Generating'
        if start_at_data:
            x = train.next_simple(batch_size=n_samples)
        else:
            x = None
        samples = generate(model_file, x=x, n_steps=100, n_samples=n_samples)

    x_v = valid.next_simple(batch_size=n_samples/10)

    print 'Finding best sigma by grid search'
    best = -2000.
    last = -3000.
    best_sigma = None

    def frange(x, y, jump):
        while x < y:
          yield x
          x += jump

    sigma = 0.15
    while True:
        print 'Sigma = %.2f' % sigma
        parzen = lep.theano_parzen(samples, sigma)
        test_ll = lep.get_ll(x_v, parzen)
        best_ll = np.mean(test_ll)
        if best_ll > best:
            best_sigma = sigma
            best = best_ll
        if best_ll < last:
            break
        last = best_ll
        sigma += 0.01

    print 'Best ll and sigma at validation: %.2f and %.2f' % (best, best_sigma)
    sigma = best_sigma

    x_t = test.next_simple(batch_size=n_samples)
    print 'Calculating log likelihood at test by Parzen window'
    parzen = lep.theano_parzen(samples, sigma)
    test_ll = lep.get_ll(x_t, parzen)
    print "Mean Log-Likelihood of test set = %.5f (model)" % np.mean(test_ll)
    print "Std of Mean Log-Likelihood of test set = %.5f" % (np.std(test_ll) / 100)
    print 'Calculating log likelihood at test by Parzen window using mnist validation data'

def unpack_rbm(dim_h=None,
               dataset='mnist',
               dataset_args=None,
               **model_args):

    dim_h = int(dim_h)

    if dataset == 'mnist':
        dim_in = 28 * 28
    elif dataset == 'horses':
        dims = dataset_args['dims']
        dim_in = dims[0] * dims[1]
    else:
        raise ValueError()

    rbm = RBM(dim_in, dim_h)
    models = [rbm]

    return models, model_args, dict(
        dataset_args=dataset_args
    )

def unpack(mode=None,
           dim_h=None,
           h_init=None,
           mlp_a=None, mlp_b=None, mlp_o=None, mlp_c=None,
           dataset='horses',
           dataset_args=None,
           **model_args):

    dataset_args = dataset_args[()]

    if mlp_a is not None:
        mlp_a = mlp_a[()]
    if mlp_b is not None:
        mlp_b = mlp_b[()]
    if mlp_o is not None:
        mlp_o = mlp_o[()]
    if mlp_c is not None:
        mlp_c = mlp_c[()]

    trng = RandomStreams(random.randint(0, 100000))

    if dataset == 'mnist':
        dim_in = 28 * 28
    elif dataset == 'horses':
        dims = dataset_args['dims']
        dim_in = dims[0] * dims[1]
    else:
        raise ValueError()

    def load_mlp(name, dim_in, dim_out,
                 dim_h=None, n_layers=None,
                 **kwargs):
        out_act = 'T.tanh'
        mlp = MLP(dim_in, dim_h, dim_out, n_layers, name=name, **kwargs)
        return mlp

    if mlp_a is not None:
        MLPa = load_mlp('MLPa', dim_in, 2 * dim_h, **mlp_a)
    else:
        MLPa = None
    if mlp_b is not None:
        MLPb = load_mlp('MLPb', dim_in, dim_h, **mlp_b)
    else:
        MLPb = None
    if mlp_o is not None:
        MLPo = load_mlp('MLPo', dim_h, dim_in, **mlp_o)
    else:
        MLPo = None
    if mlp_c is not None:
        MLPc = load_mlp('MLPc', dim_in, dim_in, **mlp_c)
    else:
        MLPc = None

    if mode == 'rnn':
        MLPa = MLPb
        MLPb = None

    if mode == 'gru':
        rnn = GenGRU(dim_in, dim_h, MLPa=MLPa, MLPb=MLPb, MLPo=MLPo, MLPc=MLPc)
        models = [rnn, rnn.MLPa, rnn.MLPb, rnn.MLPo]
    elif mode == 'rnn':
        rnn = GenRNN(dim_in, dim_h, MLPa=MLPa, MLPo=MLPo, MLPc=MLPc)
        models = [rnn, rnn.MLPa, rnn.MLPo]
    else:
        raise ValueError('Mode %s not recognized' % mode)

    if mlp_c is not None:
        models.append(rnn.MLPc)

    if h_init == 'average':
        averager = Averager((batch_size, dim_h))
        models.append(averager)
    elif h_init == 'mlp':
        mlp = MLP(dim_in, dim_h, dim_h, 1, out_act='T.tanh', name='MLPh')
        models.append(mlp)

    return models, model_args, dict(
        mode=mode,
        h_init=h_init,
        dataset=dataset,
        dataset_args=dataset_args
    )

def load_model_for_sampling(model_file):
    models, kwargs = load_model(model_file, unpack)
    dataset_args = kwargs['dataset_args']
    dataset = kwargs['dataset']

    if dataset == 'mnist':
        train = MNIST_Chains(batch_size=1, mode='train', **dataset_args)
        test = MNIST_Chains(batch_size=1, mode='test', **dataset_args)
    elif dataset == 'horses':
        train = Horses(batch_size=1, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    rnn = models['gen_{mode}'.format(mode=kwargs['mode'])]

    h_init = kwargs['h_init']
    if h_init == 'average':
        averager = models['averager']
        h0 = averager.params['m']
        f_h0 = lambda x: h0
    elif h_init == 'mlp':
        X0 = T.matrix('x0', dtype=floatX)
        mlp = models['MLPh']
        mlp.set_tparams()
        f_init = theano.function([X0], mlp(X0))
        f_h0 = lambda x: f_init(x)
    elif h_init == 'noise':
        X0 = T.matrix('X0', dtype=floatX)
        avg = T.alloc(0., X0.shape[0], rnn.dim_h).astype(floatX)

        h0 = rnn.trng.normal(size=avg.shape, avg=avg, std=0.1, dtype=avg.dtype)
        f_init = theano.function([X0], h0)
        f_h0 = lambda x: f_init(x)
    else:
        h0 = np.zeros((1, rnn.dim_h)).astype('float32')
        f_h0 = lambda x: h0

    tparams = rnn.set_tparams()
    train.set_f_energy(energy_function, rnn)
    test.set_f_energy(energy_function, rnn)

    return rnn, train, test, f_h0

def rbm_energy(model_file, rbm_file, n_samples=1000):
    samples = generate_samples(model_file, n_samples=n_samples)
    models, kwargs = load_model(rbm_file, unpack_rbm)
    rbm = models['rbm']
    rbm.set_tparams()

    X = T.tensor3('x', dtype=floatX)
    rval, updates = rbm.energy(X)
    f_energy = theano.function([X], rval['acc_neg_log_p'], updates=updates)
    rnn_energy = f_energy(samples.astype(floatX))[0]

    X0 = T.matrix('x0', dtype=floatX)
    rval, updates = rbm(n_samples, x0=X0)
    rbm_samples = rval['p']
    rval, updates_2 = rbm.energy(rbm_samples)
    updates.update(updates_2)
    f_energy = theano.function([X0], rval['acc_neg_log_p'], updates=updates)
    rbm_energy = f_energy(samples[0])[0]

    return rnn_energy, rbm_energy

def test_mixture(model_file, length=10, n_steps=100, n_samples=100):
    rnn, dataset, test, f_h0 = load_model_for_sampling(model_file)
    samples = generate_samples(model_file, n_steps=n_steps, n_samples=n_samples)[1:]
    energies = []
    r_energies = []

    X = T.tensor3('x', dtype=floatX)
    H0 = T.matrix('h0', dtype=floatX)
    f_energy = theano.function([X, H0], rnn.energy(X, h0=H0))

    samples = samples[:(n_steps // length) * length]
    r_idx = range(n_steps)
    random.shuffle(r_idx)
    s = samples.reshape(
        (length, n_steps // length * n_samples, samples.shape[2])).astype(floatX)
    r = samples[r_idx].reshape(
        (length, n_steps.shape[0] // length * n_samples, samples.shape[2])).astype(floatX)

    energy = f_energy(s, f_h0(s[0]))
    r_energy = f_energy(r, f_h0(r[0]))

    print energy.mean()
    print r_energy.mean()
    print np.exp(-energy.mean() + r_energy.mean())
    dataset.save_images(s, '/Users/devon/tmp/test_samples.png')
    dataset.save_images(r, '/Users/devon/tmp/test_rrsamples.png')

def get_sample_cross_correlation(model_file, n_steps=100):
    rnn, dataset, test, f_h0 = load_model_for_sampling(model_file)
    samples = generate_samples(model_file, n_steps=n_steps, n_samples=1)[1:, 0]

    c = np.corrcoef(samples, samples)[:n_steps, n_steps:]
    plt.imshow(c)
    plt.colorbar()
    plt.show()

def fill_in_the_blank(model_file, n_steps=200, n_samples=40, repeat=1, out_path=None):
    rnn, train, test, f_h0 = load_model_for_sampling(model_file)
    params = rnn.get_sample_params()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)
    h_s, x_s, p_s = rnn.step_sample(H, X, *params)
    f_sam = theano.function([X, H], [x_s, h_s, p_s])

    x = test.next_simple(batch_size=n_samples)
    x = np.zeros((repeat, n_samples, x.shape[1])).astype(floatX) + x[None, :, :]
    x = x.reshape((repeat * n_samples, x.shape[2]))
    ps = [np.copy(x)]
    #x[:, x.shape[1] //  2:] = 0
    #ps.append(x)
    h = f_h0(x).astype(floatX)
    x = np.zeros_like(x)
    #h = f_h0(rnn.rng.binomial(p=0.5, size=(n_samples * repeat, rnn.dim_in), n=1).astype(floatX))
    for s in xrange(n_steps):
        x, h, p = f_sam(x, h)
        ps.append(p)
        if s % 30 == 0 and s != 0:
            h = np.zeros_like(h) + f_h0(x)[0][None, :]
            #h = np.zeros_like(h) + h[0][None, :]
            #x = np.zeros_like(x)
            ps.append(np.zeros_like(x))

    train.save_images(np.array(ps), path.join(out_path, 'occulation_samples.png'))

def test_AIS(model_file, n_samples=100, M=100, K=10, f_steps=10, T_steps=10):
    models, kwargs = load_model(model_file, unpack)
    dataset_args = kwargs['dataset_args']
    dataset = kwargs['dataset']

    if dataset == 'mnist':
        test = MNIST_Chains(batch_size=n_samples, mode='test', **dataset_args)
        test.randomize()
    elif dataset == 'horses':
        test = Horses(batch_size=1, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    rnn, dataset, f_h0 = load_model_for_sampling(model_file)
    dataset.randomize()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)

    x_t = dataset.next_simple(batch_size=n_samples)

    out_f, updates_f = rnn.sample(x0=X, h0=H, n_steps=f_steps)
    f_p = theano.function([X, H], out_f['p'][-1], updates=updates_f)

    out_t, updates_t = rnn.sample(x0=X, h0=H, n_steps=T_steps)
    f_trans = theano.function([X, H], out_t['p'][-1], updates=updates_t)

    betas = [t / float(K - 1) for t in range(K)]

    x0 = dataset.next_simple(batch_size=M)

    def log_px(x, p):
        return (x * np.log(p + 1e-7) + (1 - x) * np.log(1 - p + 1e-7)).sum(axis=1)

    x_t = np.zeros((M,) + x_t.shape) + x_t[None, :, :]
    x_t = x_t.reshape((M * n_samples, x_t.shape[2]))
    p = np.zeros(x_t.shape).astype(floatX) + 0.5
    logf_ = 0. * log_px(x_t, p)
    x_k = rnn.rng.binomial(p=p, size=p.shape, n=1).astype(floatX)
    logw = np.zeros((M * n_samples,)).astype(floatX)

    for beta in betas[1:]:
        #print beta
        p = f_p(x_k, f_h0(x_k))
        logf = beta * log_px(x_t, p)
        logw = logw + logf - logf_
        print logf.mean(), logf_.mean(), logw.mean()
        trans = f_trans(x_k, f_h0(x_k)) ** beta
        x_k = rnn.rng.binomial(p=trans, size=trans.shape, n=1).astype(floatX)
        p = f_p(x_k, f_h0(x_k))
        logf_ = beta * log_px(x_t, p)

    logw = log_mean_exp(logw, axis=0, as_numpy=True)
    print logw.mean()

def generate(model_file, x=None, n_steps=20, n_samples=40, out_path=None,
             from_data=False):
    rnn, train, test, f_h0 = load_model_for_sampling(model_file)
    params = rnn.get_sample_params()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)
    h_s, x_s, p_s = rnn.step_sample(H, X, *params)
    f_sam = theano.function([X, H], [x_s, h_s, p_s])

    if x is None:
        if from_data == True:
            train.randomize()
            x = train.next_simple(batch_size=n_samples)
        else:
            x = rnn.rng.binomial(p=0.5, size=(n_samples, rnn.dim_in), n=1).astype(floatX)

    ps = [x]
    h = f_h0(x)
    for s in xrange(n_steps):
        x, h, p = f_sam(x, h)
        ps.append(p)

    if out_path is not None:
        if from_data:
            out_file = 'generation_samples(from_data).png'
        else:
            out_file = 'generation_samples.png'
        train.save_images(np.array(ps), path.join(out_path, out_file))
    else:
        return ps[-1]

def generate_samples(model_file, n_steps=1000, n_samples=1):
    rnn, dataset, test, f_h0 = load_model_for_sampling(model_file)

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)

    out_s, updates_s = rnn.sample(x0=X, h0=H, n_steps=n_steps)
    f_sample = theano.function([X, H], out_s['p'], updates=updates_s)

    x = dataset.next_simple(batch_size=n_samples)
    h = f_h0(x)
    sample_chain = f_sample(x, h)
    return sample_chain

def test_chain_likelihood(model_file, n_samples=10):
    rnn, train, test, f_h0 = load_model_for_sampling(model_file)
    params = rnn.get_sample_params()

    train.next()
    test.next()

    x_tr = train._load_chains()
    x_te = test._load_chains()

    X_tr = T.tensor3('x_tr', dtype=floatX)
    X_te = T.tensor3('x_te', dtype=floatX)
    H_tr = T.matrix('h_tr', dtype=floatX)
    H_te = T.matrix('h_te', dtype=floatX)

    outs_tr, _ = rnn(X_tr[:-1], H_tr)
    p_tr = outs_tr['p']

    outs_te, _ = rnn(X_te[:-1], H_te)
    p_te = outs_te['p']

    energy_tr = -(X_tr[1:] * T.log(p_tr + 1e-7) + (1 - X_tr[1:]) * T.log(1 - p_tr + 1e-7)).sum(axis=2).mean()
    energy_te = -(X_te[1:] * T.log(p_te + 1e-7) + (1 - X_te[1:]) * T.log(1 - p_te + 1e-7)).sum(axis=2).mean()

    f_energy = theano.function([X_tr, X_te, H_tr, H_te], [energy_tr, energy_te])

    print 'Train / Test chain energy: %s' % f_energy(x_tr, x_te, f_h0(x_tr[0]), f_h0(x_te[0]))

def visualize(model_file, out_path=None, interval=1, n_samples=-1,
              save_movie=True, use_data_every=50, use_data_in=False,
              save_hiddens=False):
    rnn, train, test, f_h0 = load_model_for_sampling(model_file)
    params = rnn.get_sample_params()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)
    h_s, x_s, p_s = rnn.step_sample(H, X, *params)
    f_sam = theano.function([X, H], [x_s, h_s, p_s])
    ps = []
    xs = []
    hs = []

    try:
        x = train.X[:1]
        h = f_h0(x)
        s = 0
        while True:
            stdout.write('\rSampling (%d): Press ^c to stop' % s)
            stdout.flush()
            x, h, p = f_sam(x, h)
            hs.append(h)
            xs.append(x)
            if use_data_every > 0 and s % use_data_every == 0:
                x_n = train.next_simple(20)
                energies, _, h_p = train.f_energy(x_n, x, h)
                energies = energies[0]
                x = x_n[np.argmin(energies)][None, :]
                if use_data_in:
                    ps.append(x)
                else:
                    ps.append(p)
            else:
                ps.append(p)

            s += 1
            if n_samples != -1 and s > n_samples:
                raise KeyboardInterrupt()
    except KeyboardInterrupt:
        print 'Finishing'

    if out_path is not None:
        train.save_images(np.array(ps), path.join(out_path, 'vis_samples.png'), x_limit=100)
        if save_hiddens:
            np.save(path.join(out_path, 'hiddens.npy'), np.array(hs))

    fig = plt.figure()
    data = np.zeros(train.dims)
    im = plt.imshow(data, vmin=0, vmax=1, cmap='Greys_r')

    def init():
        im.set_data(np.zeros(train.dims))

    def animate(i):
        data = ps[i].reshape(train.dims)
        im.set_data(data)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=s,
                                   interval=interval)

    if out_path is not None and save_movie:
        print 'Saving movie'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Devon Hjelm'), bitrate=1800)
        anim.save(path.join(out_path, 'vis_movie.mp4'), writer=writer)
    else:
        print 'Showing movie'
        plt.show()

    if out_path is not None and save_movie:
        train.next()
        fig = plt.figure()
        data = np.zeros(train.dims)
        X_tr = train._load_chains()
        im = plt.imshow(data, vmin=0, vmax=1, cmap='Greys_r')

        def animate_training_examples(i):
            data = X_tr[i, 0].reshape(train.dims)
            im.set_data(data)
            return im

        def init():
            im.set_data(np.zeros(train.dims))

        anim = animation.FuncAnimation(fig, animate_training_examples,
                                       init_func=init, frames=X_tr.shape[0],
                                       interval=interval)

        print 'Saving data movie'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Devon Hjelm'), bitrate=1800)
        anim.save(path.join(out_path, 'vis_train_movie.mp4'), writer=writer)

def energy_function(model):
    x = T.matrix('x', dtype=floatX)
    x_p = T.matrix('x_p', dtype=floatX)
    h_p = T.matrix('h_p', dtype=floatX)
    x_e = T.alloc(0., x_p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + x[None, :, :]

    params = model.get_sample_params()
    h, x_s, p = model.step_sample(h_p, x_p, *params)

    p = T.alloc(0., p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + p[:, None, :]

    energy = -(x_e * T.log(p + 1e-7) + (1 - x_e) * T.log(1 - p + 1e-7)).sum(axis=2)

    return theano.function([x, x_p, h_p], [energy, x_s, h])

def euclidean_distance(model):
    '''
    h_p are dummy variables to keep it working for dataset chain generators.
    '''

    x = T.matrix('x', dtype=floatX)
    x_p = T.matrix('x_p', dtype=floatX)
    h_p = T.matrix('h_p', dtype=floatX)
    x_e = T.alloc(0., x_p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + x[None, :, :]
    x_pe = T.alloc(0., x_p.shape[0], x.shape[0], x_p.shape[1]).astype(floatX) + x_p[:, None, :]

    params = model.get_sample_params()
    distance = (x_e - x_pe) ** 2
    distance = distance.sum(axis=2)
    return theano.function([x, x_p, h_p], [distance, x, h_p])

def random_distance(model):
    x = T.matrix('x', dtype=floatX)
    x_p = T.matrix('x_p', dtype=floatX)
    h_p = T.matrix('h_p', dtype=floatX)

    distance = model.trng.uniform(size=(x_p.shape[0], x.shape[0]), dtype=x_p.dtype)
    return theano.function([x, x_p, h_p], [distance, x, h_p])

def train_model(save_graphs=False, out_path='', name='',
                load_last=False, model_to_load=None, save_images=True,
                source=None,
                learning_rate=0.01, optimizer='adam', batch_size=10, steps=1000,
                mode='gru',
                metric='energy',
                dim_h=500,
                mlp_a=None, mlp_b=None, mlp_o=None, mlp_c=None,
                dataset=None, dataset_args=None,
                noise_input=True, sample=True,
                h_init='mlp',
                model_save_freq=100, show_freq=10):

    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    window = dataset_args['window']
    stride = min(window, dataset_args['chain_stride'])
    out_path = path.abspath(out_path)

    if dataset == 'mnist':
        train = MNIST_Chains(batch_size=batch_size, out_path=out_path, **dataset_args)
    elif dataset == 'horses':
        train = Horses(batch_size=batch_size, out_path=out_path, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    dim_in = train.dim
    X = T.tensor3('x', dtype=floatX)
    trng = RandomStreams(random.randint(0, 100000))

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError()

    print 'Forming model'

    def load_mlp(name, dim_in, dim_out,
                 dim_h=None, n_layers=None,
                 **kwargs):
        out_act = 'T.tanh'
        mlp = MLP(dim_in, dim_h, dim_out, n_layers, **kwargs)
        return mlp

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack)
    else:
        mlps = {}
        if mode == 'gru':
            if mlp_a is not None:
                MLPa = load_mlp('MLPa', dim_in, 2 * dim_h, **mlp_a)
            else:
                MLPa = None
            mlps['MLPa'] = MLPa

        if mlp_b is not None:
            MLPb = load_mlp('MLPb', dim_in, dim_h, **mlp_b)
        else:
            MLPb = None
        if mode == 'gru':
            mlps['MLPb'] = MLPb
        else:
            mlps['MLPa'] = MLPb

        if mlp_o is not None:
            MLPo = load_mlp('MLPo', dim_h, dim_in, **mlp_o)
        else:
            MLPo = None
        mlps['MLPo'] = MLPo

        if mlp_c is not None:
            MLPc = load_mlp('MLPc', dim_in, dim_in, **mlp_c)
        else:
            MLPc = None
        mlps['MLPc'] = MLPc

        rnn = C(dim_in, dim_h, trng=trng,
                **mlps)
        models = OrderedDict()
        models[rnn.name] = rnn

    print 'Getting params...'
    rnn = models['gen_{mode}'.format(mode=mode)]
    tparams = rnn.set_tparams()

    X = trng.binomial(p=X, size=X.shape, n=1, dtype=X.dtype)
    X_s = X[:-1]
    updates = theano.OrderedUpdates()
    if noise_input:
        print 'Noising input'
        X_s = X_s * (1 - trng.binomial(p=0.1, size=X_s.shape, n=1, dtype=X_s.dtype))

    if h_init is None:
        h0 = None
    elif h_init == 'last':
        print 'Initializing h0 from chain'
        h0 = theano.shared(np.zeros((batch_size, rnn.dim_h)).astype(floatX))
    elif h_init == 'noise':
        print 'Initializing h0 from noise'
        h0 = trng.normal(avg=0, std=0.1, size=(batch_size, rnn.dim_h)).astype(floatX)
    elif h_init == 'average':
        print 'Initializing h0 from running average'
        if 'averager' in models.keys():
            'Found pretrained averager'
            averager = models['averager']
        else:
            averager = Averager((dim_h))
        tparams.update(averager.set_tparams())
        h0 = (T.alloc(0., batch_size, rnn.dim_h) + averager.m[None, :]).astype(floatX)
    elif h_init == 'mlp':
        print 'Initializing h0 from MLP'
        if 'MLPh' in models.keys():
            print 'Found pretrained MLP'
            mlp = models['MLPh']
        else:
            mlp = MLP(rnn.dim_in, rnn.dim_h, rnn.dim_h, 1,
                      out_act='T.tanh',
                      name='MLPh')
        tparams.update(mlp.set_tparams())
        h0s = mlp(D)
        h0 = h0s[0]

    print 'Model params: %s' % tparams.keys()
    if metric == 'energy':
        print 'Energy-based metric'
        train.set_f_energy(energy_function, rnn)
    elif metric in ['euclidean', 'euclidean_then_energy']:
        print 'Euclidean-based metic'
        train.set_f_energy(euclidean_distance, rnn)
    else:
        raise ValueError(metric)

    outs, updates_1 = rnn(X_s, h0=h0)
    h = outs['h']
    p = outs['p']
    x = outs['y']
    updates.update(updates_1)

    energy = -(X[1:] * T.log(p + 1e-7) + (1 - X[1:]) * T.log(1 - p + 1e-7)).sum(axis=(0, 2))
    cost = energy.mean()
    consider_constant = [x]

    if h_init == 'last':
        updates += [(h0, h[stride - 1])]
    elif h_init == 'average':
        outs_h, updates_h = averager(h)
        updates.update(updates_h)
    elif h_init == 'mlp':
        h_c = T.zeros_like(h) + h
        cost += ((h0s - h_c)**2).sum(axis=2).mean()
        consider_constant.append(h_c)

    extra_outs = [energy.mean(), h, p]

    if sample:
        print 'Setting up sampler'
        if h_init == 'average':
            h0_s = T.alloc(0., window, rnn.dim_h).astype(floatX) + averager.m[None, :]
        elif h_init == 'mlp':
            h0_s = mlp(X[:, 0])
        elif h_init == 'noise':
            h0_s = trng.normal(avg=0, std=0.1, size=(window, rnn.dim_h)).astype(floatX)
        elif h_init == 'last':
            h0_s = h[:, 0]
        else:
            h0_s = None
        out_s, updates_s = rnn.sample(X[:, 0], h0=h0_s, n_samples=10, n_steps=10)
        f_sample = theano.function([X], out_s['p'], updates=updates_s)

    grad_tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(grad_tparams),
                   consider_constant=consider_constant)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=extra_outs)

    print 'Actually running'

    try:
        e = 0
        for s in xrange(steps):
            try:
                x, _ = train.next()
            except StopIteration:
                e += 1
                print 'Epoch {epoch}'.format(epoch=e)
                if metric == 'euclidean_then_energy' and e == 2:
                    print 'Switching to model energy'
                    train.set_f_energy(energy_function, rnn)
                continue
            rval = f_grad_shared(x)

            if check_bad_nums(rval, ['cost', 'energy', 'h', 'x', 'p']):
                return

            if s % show_freq == 0:
                print ('%d: cost: %.5f | energy: %.2f | prob: %.2f'
                       % (e, rval[0], rval[1], np.exp(-rval[1])))
            if s % model_save_freq == 0:
                idx = np.random.randint(rval[3].shape[1])
                samples = np.concatenate([x[1:, idx, :][None, :, :],
                                        rval[3][:, idx, :][None, :, :]], axis=0)
                train.save_images(
                    samples,
                    path.join(
                        out_path,
                        '{name}_inference_chain.png'.format(name=name)))
                train.save_images(
                    x, path.join(
                        out_path, '{name}_input_samples.png'.format(name=name)))
                if sample:
                    sample_chain = f_sample(x)
                    train.save_images(
                        sample_chain,
                        path.join(
                            out_path, '{name}_samples.png'.format(name=name)))

                temp_file = path.join(
                    out_path, '{name}_temp.npz'.format(name=name))
                d = dict((k, v.get_value()) for k, v in tparams.items())
                d.update(mode=mode,
                         dim_h=dim_h,
                         h_init=h_init,
                         mlp_a=mlp_a, mlp_b=mlp_b, mlp_o=mlp_o, mlp_c=mlp_c,
                         dataset=dataset, dataset_args=dataset_args)
                np.savez(temp_file, **d)

            f_grad_updates(learning_rate)
    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = os.path.join(
        out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
    last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

    print 'Saving the following params: %s' % tparams.keys()
    d = dict((k, v.get_value()) for k, v in tparams.items())
    d.update(mode=mode,
             dim_h=dim_h,
             h_init=h_init,
             mlp_a=mlp_a, mlp_b=mlp_b, mlp_o=mlp_o, mlp_c=mlp_c,
             dataset=dataset, dataset_args=dataset_args)

    np.savez(outfile, **d)
    np.savez(last_outfile,  **d)
    print 'Done saving. Bye bye.'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    return parser

def load_experiment(experiment_yaml):
    print('Loading experiment from %s' % experiment_yaml)
    exp_dict = yaml.load(open(experiment_yaml))
    print('Experiment hyperparams: %s' % pprint.pformat(exp_dict))
    return exp_dict

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    out_path = path.join(args.out_path, exp_dict['name'])

    if out_path is not None:
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)