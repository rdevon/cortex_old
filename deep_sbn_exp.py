'''
SFFN experiment
'''

import argparse
from collections import OrderedDict
from glob import glob
from monitor import SimpleMonitor
import numpy as np
import os
from os import path
import pprint
from progressbar import ProgressBar
import random
import shutil
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from layers import MLP
from mnist import MNIST
import op
from gbn import DeepGBN as DGBN
from sbn import DeepSBN as DSBN
from tools import (
    check_bad_nums,
    itemlist,
    load_model,
    load_experiment,
    _slice
)

floatX = theano.config.floatX

def concatenate_inputs(model, y, py):
    '''
    Function to concatenate ground truth to samples and probabilities.
    '''
    y_hat = model.conditionals[0].sample(py)

    py = T.concatenate([y[None, :, :], py], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    return py, y

def load_mlp(name, dim_in, dim_out, dim_h=None, n_layers=None, **kwargs):
    mlp = MLP(dim_in, dim_h, dim_out, n_layers, name=name, **kwargs)
    return mlp

def load_data(dataset,
              train_batch_size,
              valid_batch_size,
              test_batch_size,
              **dataset_args):
    if dataset == 'mnist':
        if train_batch_size is not None:
            train = MNIST(batch_size=train_batch_size,
                          mode='train',
                          inf=False,
                          **dataset_args)
        else:
            train = None
        if valid_batch_size is not None:
            valid = MNIST(batch_size=valid_batch_size,
                          mode='valid',
                          inf=False,
                          **dataset_args)
        else:
            valid = None
        if test_batch_size is not None:
            test = MNIST(batch_size=test_batch_size,
                         mode='test',
                         inf=False,
                         **dataset_args)
        else:
            test = None
    else:
        raise ValueError()

    return train, valid, test

def unpack(dim_h=None,
           z_init=None,
           recognition_net=None,
           generation_net=None,
           prior=None,
           n_layers=None,
           dataset=None,
           dataset_args=None,
           n_inference_steps=None,
           inference_method=None,
           inference_rate=None,
           entropy_scale=None,
           input_mode=None,
           **model_args):
    '''
    Function to unpack pretrained model into fresh SFFN class.
    '''

    kwargs = dict(
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_steps=n_inference_steps,
        z_init=z_init
    )

    n_layers = int(n_layers)
    dim_h = int(dim_h)
    dataset_args = dataset_args[()]

    if dataset == 'mnist':
        dim_in = 28 * 28
        dim_out = dim_in
    else:
        raise ValueError()

    if prior == 'logistic':
        out_act = 'T.nnet.sigmoid'
    elif prior == 'gaussian':
        out_act = 'lambda x: x'
    else:
        raise ValueError('%s prior not known' % prior)

    models = []
    if recognition_net is not None:
        recognition_net = recognition_net[()]
        posteriors = []
        for l in xrange(n_layers):
            if l == 0:
                mlp_dim_in = dim_in
                name = 'posterior'
            else:
                mlp_dim_in = dim_h
                name = 'posterior%d' % l

            posteriors.append(load_mlp(name, mlp_dim_in, dim_h,
                                       out_act=out_act,
                                       **recognition_net))
    else:
        posteriors = None

    if generation_net is not None:
        generation_net = generation_net[()]
        conditionals = []
        for l in xrange(n_layers):
            if l == 0:
                mlp_dim_out = dim_in
                name = 'conditional'
            else:
                mlp_dim_out = dim_h
                name = 'conditional%d' % l

            conditionals.append(load_mlp(name, dim_h, mlp_dim_out,
                                         out_act='T.nnet.sigmoid',
                                         **generation_net))
    else:
        conditionals = None

    if prior == 'logistic':
        C = DSBN
    elif prior == 'gaussian':
        C = DGBN
    else:
        raise ValueError()
    model = C(dim_in, dim_h, dim_out, n_layers=n_layers,
            conditionals=conditionals,
            posteriors=posteriors,
            **kwargs)
    models.append(model)

    return models, model_args, dict(
        z_init=z_init,
        dataset=dataset,
        dataset_args=dataset_args
    )

def train_model(
    out_path='', name='', load_last=False, model_to_load=None, save_images=True,

    learning_rate=0.1, optimizer='adam',
    batch_size=100, valid_batch_size=100, test_batch_size=1000,
    max_valid=10000,
    epochs=100,

    dim_h=300, prior='logistic',
    n_layers=2,
    input_mode=None,
    generation_net=None, recognition_net=None,
    excludes=['log_sigma'],
    center_input=True, center_latent=False,

    z_init=None,
    inference_method='momentum',
    inference_rate=.01,
    n_inference_steps=100,
    n_inference_steps_test=0,
    inference_decay=1.0,
    n_inference_samples=20,
    inference_scaling=None,
    entropy_scale=1.0,
    alpha=7,
    n_sampling_steps=0,
    n_sampling_steps_test=0,

    n_mcmc_samples=20,
    n_mcmc_samples_test=20,
    importance_sampling=False,

    dataset=None, dataset_args=None,
    model_save_freq=1000, show_freq=100, archive_every=0
    ):

    kwargs = dict(
        prior=prior,
        n_layers=n_layers,
        z_init=z_init,
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_samples=n_inference_samples
    )

    # ========================================================================
    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Model args: %s' % pprint.pformat(kwargs)

    # ========================================================================
    print 'Setting up data'
    train, valid, test = load_data(dataset,
                                   batch_size,
                                   valid_batch_size,
                                   test_batch_size,
                                   **dataset_args)

    # ========================================================================
    print 'Setting model and variables'
    dim_in = train.dim
    dim_out = train.dim
    X = T.matrix('x', dtype=floatX)
    X.tag.test_value = np.zeros((batch_size, 784), dtype=X.dtype)
    trng = RandomStreams(random.randint(0, 1000000))

    if input_mode == 'sample':
        print 'Sampling datapoints'
        X = trng.binomial(p=X, size=X.shape, n=1, dtype=X.dtype)
    elif input_mode == 'noise':
        print 'Adding noise to data points'
        X = X * trng.binomial(p=0.1, size=X.shape, n=1, dtype=X.dtype)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print 'Loading model and forming graph'

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack, **kwargs)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack, **kwargs)
    else:
        if prior == 'logistic':
            out_act = 'T.nnet.sigmoid'
        elif prior == 'gaussian':
            out_act = 'lambda x: x'
        else:
            raise ValueError('%s prior not known' % prior)

        if recognition_net is not None:
            posteriors = []
            for l in xrange(n_layers):
                if l == 0:
                    mlp_dim_in = dim_in
                else:
                    mlp_dim_in = dim_h
                posteriors.append(load_mlp('posterior', mlp_dim_in, dim_h,
                                           out_act=out_act,
                                           **recognition_net))
        else:
            posteriors = None

        if generation_net is not None:
            conditionals = []
            for l in xrange(n_layers):
                if l == 0:
                    mlp_dim_out = dim_in
                else:
                    mlp_dim_out = dim_h
                conditionals.append(load_mlp('conditional', dim_h, mlp_dim_out,
                                             out_act='T.nnet.sigmoid',
                                             **generation_net))
        else:
            conditionals = None

        if prior == 'logistic':
            C = DSBN
        elif prior == 'gaussian':
            C = DGBN
        else:
            raise ValueError()
        model = C(dim_in, dim_h, dim_out, trng=trng,
                conditionals=conditionals,
                posteriors=posteriors,
                **kwargs)

        models = OrderedDict()
        models[model.name] = model

    if prior == 'logistic':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    tparams = model.set_tparams(excludes=excludes)

    # ========================================================================
    print 'Getting cost'
    (zss, prior_energy, h_energy, y_energy), updates, constants = model.inference(
        X_i, X, n_inference_steps=n_inference_steps,
        n_sampling_steps=n_sampling_steps, n_samples=n_mcmc_samples)

    cost = prior_energy + h_energy + y_energy

    # ========================================================================
    print 'Extra functions'

    # Test function with sampling
    rval, updates_s = model(
        X_i, X, n_samples=n_mcmc_samples_test, n_inference_steps=n_inference_steps_test,
        n_sampling_steps=n_sampling_steps_test)

    py_s = rval['py']
    lower_bound = rval['lower_bound']
    pd_s, d_hat_s = concatenate_inputs(model, X_i, py_s)

    outs_s = [lower_bound, pd_s, d_hat_s]
    outs_s.append(rval['lower_bound_gain'])

    f_test = theano.function([X], outs_s, updates=updates_s)

    # Sample from prior
    py_p = model.sample_from_prior()
    f_prior = theano.function([], py_p)

    # ========================================================================

    extra_outs = [prior_energy, h_energy, y_energy]

    extra_outs_names = ['cost', '-log p(h_n)', 'KL(q||q~)',
                        '-log p(x|h)']

    # ========================================================================
    print 'Setting final tparams and save function'

    all_params = OrderedDict((k, v) for k, v in tparams.iteritems())

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    print 'Learned model params: %s' % tparams.keys()
    print 'Saved params: %s' % all_params.keys()

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())

        d.update(
            dim_h=dim_h,
            input_mode=input_mode,
            prior=prior,
            generation_net=generation_net, recognition_net=recognition_net,
            dataset=dataset, dataset_args=dataset_args
        )
        np.savez(outfile, **d)

    # ========================================================================
    print 'Getting gradients.'
    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=constants)

    # ========================================================================
    print 'Building optimizer'
    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=extra_outs)

    monitor = SimpleMonitor()

    # ========================================================================
    print 'Actually running'

    best_cost = float('inf')
    best_epoch = 0

    valid_lbs = []
    train_lbs = []

    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

    try:
        t0 = time.time()
        s = 0
        e = 0
        while True:
            try:
                x, _ = train.next()
            except StopIteration:
                print 'End Epoch {epoch} ({name})'.format(epoch=e, name=name)
                print '=' * 100
                valid.reset()

                lb_vs = []
                lb_ts = []

                print 'Validating'
                pbar = ProgressBar(maxval=min(max_valid, valid.n)).start()
                while True:
                    try:
                        if valid.pos != -1:
                            pbar.update(valid.pos)

                        x_v, _ = valid.next()
                        x_t, _ = train.next()
                        if valid.pos >= max_valid:
                            raise StopIteration

                        lb_v = f_test(x_v)[0]
                        lb_t = f_test(x_t)[0]

                        lb_vs.append(lb_v)
                        lb_ts.append(lb_t)

                    except StopIteration:
                        break

                lb_v = np.mean(lb_vs)
                lb_t = np.mean(lb_ts)

                print 'Train / Valid lower bound at end of epoch: %.2f / %.2f' % (lb_t, lb_v)

                if lb_v < best_cost:
                    print 'Found best: %.2f' % lb_v
                    best_cost = lb_v
                    best_epoch = e
                    if out_path is not None:
                        print 'Saving best to %s' % bestfile
                        save(tparams, bestfile)
                else:
                    print 'Best (%.2f) at epoch %d' % (best_cost, best_epoch)

                valid_lbs.append(lb_v)
                train_lbs.append(lb_t)

                if out_path is not None:
                    print 'Saving lower bounds in %s' % out_path
                    np.save(path.join(out_path, 'valid_lbs.npy'), valid_lbs)
                    np.save(path.join(out_path, 'train_lbs.npy'), train_lbs)

                e += 1

                print '=' * 100
                print 'Epoch {epoch} ({name})'.format(epoch=e, name=name)
                # HACK
                #if e == 1:
                #    learning_rate = learning_rate / 10.
                #    print 'New learning rate: %.5f' % learning_rate

                valid.reset()
                train.reset()
                continue

            if e > epochs:
                break

            rval = f_grad_shared(x)

            if check_bad_nums(rval, extra_outs_names):
                return

            if s % show_freq == 0:
                try:
                    x_v, _ = valid.next()
                except StopIteration:
                    x_v, _ = valid.next()

                outs_v = f_test(x_v)
                outs_t = f_test(x)

                lb_v, pd_v, d_hat_v, lbg_v = outs_v[:4]
                lb_t = outs_t[0]

                outs = OrderedDict((k, v)
                    for k, v in zip(extra_outs_names,
                                    rval[:len(extra_outs_names)]))

                t1 = time.time()
                outs.update(**{
                    '-log p(x) <= (t)': lb_t,
                    '-log p(x) <= (v)': lb_v,
                    '-d log p(x)': lbg_v,
                    'dt': t1-t0}
                )

                monitor.update(**outs)
                t0 = time.time()

                monitor.display(e, s)

                if save_images and s % model_save_freq == 0:
                    monitor.save(path.join(
                        out_path, '{name}_monitor.png').format(name=name))
                    if archive_every and s % archive_every == 0:
                        monitor.save(path.join(
                            out_path, '{name}_monitor({s})'.format(name=name, s=s))
                        )

                    d_hat_s = np.concatenate([pd_v[:10],
                                              d_hat_v[1][None, :, :]], axis=0)
                    d_hat_s = d_hat_s[:, :min(10, d_hat_s.shape[1] - 1)]
                    train.save_images(d_hat_s, path.join(
                        out_path, '{name}_samples.png'.format(name=name)))

                    pd_p = f_prior()
                    train.save_images(
                        pd_p[:, None], path.join(
                            out_path,
                            '{name}_samples_from_prior.png'.format(name=name)),
                        x_limit=10
                    )

            f_grad_updates(learning_rate)

            s += 1

    except KeyboardInterrupt:
        print 'Training interrupted'

    if out_path is not None:
        outfile = path.join(out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
        last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

        print 'Saving'
        save(tparams, outfile)
        save(tparams, last_outfile)
        print 'Done saving.'

    print 'Bye bye!'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    parser.add_argument('-n', '--name', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    if args.name is not None:
        exp_dict['name'] = args.name
    out_path = path.join(args.out_path, exp_dict['name'])

    if out_path is not None:
        print 'Saving to %s' % out_path
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)
