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

def lower_bound(model_file, mode='gru', source=None, out_path=None,
                samples=10000, sigma=0.2):
    train = mnist_iterator(batch_size=1, mode='train', inf=False)
    test = mnist_iterator(batch_size=samples, mode='test', inf=False)
    valid = mnist_iterator(batch_size=samples / 10, mode='valid', inf=False)
    '''
    train = SimpleHorses(batch_size=1, source=source, inf=True)
    '''
    trng = RandomStreams(6 * 23 * 2015)
    dim_in = train.dim
    dim_h=200

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError()

    rnn = C(dim_in, dim_h, trng=trng, h0_mode='ffn', condition_on_x=True)
    rnn = load_model(rnn, model_file)

    tparams = rnn.set_tparams()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)

    params = rnn.get_params()
    h_s, x_s, p_s = rnn.step_sample(H, X, *params)

    f_h0 = theano.function([X, H], [x_s, h_s, p_s])

    xs = []
    x, _ = train.next()
    h = np.zeros((x.shape[0], rnn.dim_h)).astype('float32')
    while len(xs) < samples:
        x, h, p = f_h0(x, h)
        xs.append(p)

    samples = np.array(xs)[:, 0]
    x_v, _ = valid.next()

    print 'Finding best sigma by grid search'
    best = 0
    best_sigma = None

    def frange(x, y, jump):
        while x < y:
          yield x
          x += jump

    for sigma in frange(0.15, 0.25, 0.01):
        print 'Sigma = %.2f' % sigma
        parzen = lep.theano_parzen(samples, sigma)
        test_ll = lep.get_ll(x_v, parzen)
        best_ll = np.mean(test_ll)
        if best_ll > best:
            best_sigma = sigma
            best = best_ll

    print 'Best ll and sigma at validation: %.2f and %.2f' % (best, best_sigma)
    sigma = best_sigma

    x_t, _ = test.next()
    print 'Calculating log likelihood at test by Parzen window'
    parzen = lep.theano_parzen(samples, sigma)
    test_ll = lep.get_ll(x_t, parzen)
    print "Mean Log-Likelihood of test set = %.5f (model)" % np.mean(test_ll)
    print "Std of Mean Log-Likelihood of test set = %.5f" % (np.std(test_ll) / 100)
    print 'Calculating log likelihood at test by Parzen window using mnist validation data'
    #x, _ = valid.next()
    #parzen = lep.theano_parzen(x, sigma)
    #test_ll = lep.get_ll(x_t, parzen)
    #print "Mean Log-Likelihood of test set = %.5f (MNIST)" % np.mean(test_ll)
    #print "Std of Mean Log-Likelihood of test set = %.5f" % (np.std(test_ll) / 100)

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

def unpack(batch_size=None,
           dim_h=None,
           optimizer=None,
           mode=None,
           learning_rate=None,
           x_cond=None,
           steps=None,
           dataset='horses',
           dataset_args=None,
           h_init=None,
           **model_args):

    dataset_args = dataset_args[()]
    x_cond = x_cond[()]
    models = []

    trng = RandomStreams(random.randint(0, 100000))

    if dataset == 'mnist':
        dim_in = 28 * 28
    elif dataset == 'horses':
        dims = dataset_args['dims']
        dim_in = dims[0] * dims[1]
    else:
        raise ValueError()

    if x_cond is not None:
        print 'Found xcond: \n%s' % pprint.pformat(x_cond)
        condition_on_x  = MLP(dim_in=dim_in, dim_h=x_cond['dim_h'],
                              dim_out=dim_in, n_layers=x_cond['n_layers'],
                              h_act=x_cond['h_act'], name='xcond')
        models.append(condition_on_x)
    else:
        condition_on_x = None

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError('Mode %s not recognized' % mode)

    rnn = C(dim_in, dim_h, trng=trng, condition_on_x=condition_on_x)
    models.append(rnn)

    if h_init == 'average':
        averager = Averager((batch_size, dim_h))
        models.append(averager)
    elif h_init == 'mlp':
        mlp = MLP(dim_in, dim_h, dim_h, 1, out_act='T.tanh')
        models.append(mlp)

    return models, model_args, dict(
        batch_size=batch_size,
        optimizer=optimizer,
        mode=mode,
        dataset=dataset,
        learning_rate=learning_rate,
        condition_on_x=condition_on_x,
        steps=steps,
        h_init=h_init,
        dataset_args=dataset_args
    )

def load_model_for_sampling(model_file):
    models, kwargs = load_model(model_file, unpack)
    dataset_args = kwargs['dataset_args']
    dataset = kwargs['dataset']

    if dataset == 'mnist':
        train = MNIST_Chains(batch_size=1, mode='train', **dataset_args)
    elif dataset == 'horses':
        train = Horses(batch_size=1, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    rnn = models['gen_gru']

    h_init = kwargs['h_init']
    if h_init == 'average':
        averager = models['averager']
        h0 = averager.params['m']
    elif h_init == 'mlp':
        X0 = T.matrix('x0', dtype=floatX)
        mlp = models['MLP']
        mlp.set_tparams()
        f_init = theano.function([X0], mlp(X0))
        h0 = f_init(train.X[:1])
    else:
        h0 = np.zeros((1, rnn.dim_h)).astype('float32')

    tparams = rnn.set_tparams()
    train.set_f_energy(energy_function, rnn)

    return rnn, train, h0

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

def generate_samples(model_file, n_samples=1000):
    rnn, dataset, h0 = load_model_for_sampling(model_file)

    X = T.matrix('x', dtype=floatX)

    out_s, updates_s = rnn.sample(X, n_steps=n_samples)
    f_sample = theano.function([X], out_s['p'], updates=updates_s)

    idx = random.randint(0, dataset.n)
    x = dataset.X[idx][None, :]
    sample_chain = f_sample(x)
    return sample_chain

def visualize(model_file, out_path=None, interval=1, n_samples=-1, save_movie=True,
              use_data_every=50):
    rnn, train, h0 = load_model_for_sampling(model_file)
    params = rnn.get_params()

    X = T.matrix('x', dtype=floatX)
    H = T.matrix('h', dtype=floatX)
    h_s, x_s, p_s = rnn.step_sample(H, X, *params)
    f_h0 = theano.function([X, H], [x_s, h_s, p_s])
    ps = []
    xs = []

    try:
        x = train.X[:1]
        h = h0
        s = 0
        while True:
            stdout.write('\rSampling (%d): Press ^c to stop' % s)
            stdout.flush()
            x, h, p = f_h0(x, h)
            ps.append(p)
            xs.append(x)
            if use_data_every > 0 and s % use_data_every == 0:
                x_n = train.next_simple(20)
                energies, _, h_p = train.f_energy(x_n, x, h)
                energies = energies[0]
                x = x_n[np.argmin(energies)][None, :]
            s += 1
            if n_samples != -1 and s > n_samples:
                raise KeyboardInterrupt()
    except KeyboardInterrupt:
        print 'Finishing'

    if out_path is not None:
        train.save_images(np.array(ps), path.join(out_path, 'vis_samples.png'), x_limit=500)

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
        train_chain.next()
        fig = plt.figure()
        data = np.zeros(train.dims)
        X_tr = train_chain._load_chains()
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

        print 'Saving movie'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Devon Hjelm'), bitrate=1800)
        anim.save(path.join(out_path, 'vis_train_movie.mp4'), writer=writer)

def energy_function(model):
    x = T.matrix('x', dtype=floatX)
    x_p = T.matrix('x_p', dtype=floatX)
    h_p = T.matrix('h_p', dtype=floatX)

    x_e = T.alloc(0., x_p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + x[None, :, :]

    params = model.get_params()
    h, x_s, p = model.step_sample(h_p, x_p, *params)
    p = T.alloc(0., p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + p[:, None, :]

    energy = -(x_e * T.log(p + 1e-7) + (1 - x_e) * T.log(1 - p + 1e-7)).sum(axis=2)

    return theano.function([x, x_p, h_p], [energy, x_s, h])

def train_model(save_graphs=False, out_path='', name='',
                load_last=False, model_to_load=None, save_images=True,
                source=None, batch_size=1, dim_h=500, optimizer='adam', mode='gru',
                learning_rate=0.01, x_cond=None, dataset=None, steps=1000,
                dataset_args=None, noise_input=True, sample=True, h_init='mlp',
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
    condition_on_x = None
    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack)
    else:
        if x_cond is not None:
            condition_on_x = MLP(dim_in=dim_in, dim_h=x_cond['dim_h'],
                                 dim_out=dim_in, n_layers=x_cond['n_layers'],
                                 h_act=x_cond['h_act'], name='xcond')
        else:
            condition_on_x = None
        rnn = C(dim_in, dim_h, trng=trng, condition_on_x=condition_on_x)
        models = OrderedDict()
        models[rnn.name] = rnn

    print 'Getting params...'
    rnn = models['gen_gru']
    tparams = rnn.set_tparams()

    X = trng.binomial(p=X, size=X.shape, n=1, dtype=X.dtype)
    X_s = X[:-1]
    updates = theano.OrderedUpdates()
    if noise_input:
        X_s = X_s * (1 - trng.binomial(p=0.1, size=X_s.shape, n=1, dtype=X_s.dtype))

    if h_init is None:
        h0 = None
    elif h_init == 'last':
        print 'Initializing h0 from chain'
        h0 = theano.shared(np.zeros((batch_size, rnn.dim_h)).astype(floatX))
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
        if 'MLP' in models.keys():
            print 'Found pretrained MLP'
            mlp = models['MLP']
        else:
            mlp = MLP(rnn.dim_in, rnn.dim_h, rnn.dim_h, 1, out_act='T.tanh')
        tparams.update(mlp.set_tparams())
        h0 = mlp(X[0])

    print 'Model params: %s' % tparams.keys()
    train.set_f_energy(energy_function, rnn)

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
        h_c = T.zeros_like(h[0]) + h[0]
        cost += ((h0 - h[0])**2).sum(axis=1).mean()
        consider_constant.append(h_c)

    extra_outs = [energy.mean(), h, p]

    if sample:
        print 'Setting up sampler'
        if h_init == 'average':
            h0_s = T.alloc(0., window, rnn.dim_h).astype(floatX) + averager.m[None, :]
        elif h_init == 'mlp':
            h0_s = mlp(X[:, 0])
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
        for e in xrange(steps):
            x, _ = train.next()
            rval = f_grad_shared(x)

            if check_bad_nums(rval, ['cost', 'energy', 'h', 'x', 'p']):
                return

            if e % show_freq == 0:
                print ('%d: cost: %.5f | energy: %.2f | prob: %.2f'
                       % (e, rval[0], rval[1], np.exp(-rval[1])))
            if e % show_freq == 0:
                idx = np.random.randint(rval[3].shape[1])
                samples = np.concatenate([x[1:, idx, :][None, :, :],
                                        rval[3][:, idx, :][None, :, :]], axis=0)
                train.save_images(samples, path.join(out_path, 'inference_chain.png'))
                train.save_images(x, path.join(out_path, 'input_samples.png'))
                if sample:
                    sample_chain = f_sample(x)
                    train.save_images(sample_chain, path.join(out_path, 'samples.png'))
            if e % model_save_freq == 0:
                temp_file = path.join(out_path, '{name}_temp.npz'.format(name=name))
                d = dict((k, v.get_value()) for k, v in tparams.items())
                d.update(batch_size=batch_size,
                         dim_h=dim_h,
                         optimizer=optimizer,
                         mode=mode,
                         learning_rate=learning_rate,
                         x_cond=x_cond,
                         steps=steps,
                         dataset=dataset,
                         h_init=h_init,
                         dataset_args=dataset_args)
                np.savez(temp_file, **d)

            f_grad_updates(learning_rate)
    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = os.path.join(out_path,
                           '{name}_{t}.npz'.format(name=name, t=int(time.time())))
    last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

    print 'Saving the following params: %s' % tparams.keys()
    d = dict((k, v.get_value()) for k, v in tparams.items())
    d.update(batch_size=batch_size,
             dim_h=dim_h,
             optimizer=optimizer,
             mode=mode,
             learning_rate=learning_rate,
             x_cond=x_cond,
             steps=steps,
             dataset=dataset,
             h_init=h_init,
             dataset_args=dataset_args)

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

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)