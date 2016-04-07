'''
Demo for training RBM with MNIST dataset.
'''

from collections import OrderedDict
import numpy as np
from os import path
import pprint
import sys
import theano
from theano import tensor as T
import time

from datasets import load_data
from models.rbm import RBM, unpack
from utils.monitor import SimpleMonitor
from utils import floatX
from utils.tools import get_trng, print_profile, print_section
from utils.training import (
    main_loop,
    make_argument_parser,
    set_experiment,
    set_model,
    set_optimizer,
    set_params
)


def init_learning_args(
    learning_rate=0.0001,
    optimizer='sgd',
    optimizer_args=dict(),
    learning_rate_schedule=None,
    batch_size=100,
    valid_batch_size=100,
    epochs=100,
    valid_key='nll',
    valid_sign='+'):
    return locals()

def init_inference_args(
    n_chains=10,
    persistent=False,
    n_steps=1):
    return locals()

def train(
    out_path='', name='', model_to_load=None, save_images=True,
    dim_h=None, center_input=False,
    learning_args=dict(),
    inference_args=dict(),
    dataset_args=dict()):

    # ========================================================================
    learning_args = init_learning_args(**learning_args)
    inference_args = init_inference_args(**inference_args)

    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Learning args: %s' % pprint.pformat(learning_args)
    print 'Inference args: %s' % pprint.pformat(inference_args)

    # ========================================================================
    print_section('Setting up data')
    batch_size = learning_args.pop('batch_size')
    valid_batch_size = learning_args.pop('valid_batch_size')
    train, valid, test = load_data(
        train_batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        **dataset_args)

    # ========================================================================
    print_section('Setting model and variables')
    dim_in = train.dims[train.name]

    X = T.matrix('x', dtype=floatX)
    X.tag.test_value = np.zeros((batch_size, dim_in), dtype=X.dtype)
    trng = get_trng()

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print_section('Loading model and forming graph')

    def create_model():
        model = RBM(dim_in, dim_h)
        models = OrderedDict()
        models[model.name] = model
        return models

    models = set_model(create_model, model_to_load, unpack)
    model = models['rbm']
    tparams = model.set_tparams()
    print_profile(tparams)

    # ==========================================================================
    print_section('Getting cost')

    persistent = inference_args.pop('persistent')
    if persistent:
        H_p = theano.shared(
            np.zeros((inference_args['n_chains'], model.dim_h)).astype(floatX),
            name='h_p')
    else:
        H_p = None
    results, samples, updates, constants = model(
        X_i, h_p=H_p, **inference_args)

    updates = theano.OrderedUpdates()
    if persistent:
        updates += theano.OrderedUpdates([(H_p, samples['hs'][-1])])

    cost = results['cost']
    extra_outs = [results['free_energy']]
    extra_outs_keys = ['cost', 'free_energy']

    # ==========================================================================
    print_section('Test functions')
    f_test_keys = results.keys()
    f_test = theano.function([X], results.values())

    H0 = model.trng.binomial(size=(10, model.dim_h), dtype=floatX)
    s_outs, s_updates = model.sample(H0, n_steps=100)
    f_chain = theano.function(
        [], s_outs['pvs'], updates=s_updates)

     # ========================================================================
    print_section('Setting final tparams and save function')
    tparams, all_params = set_params(tparams, updates)

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())
        d.update(
            dim_in=dim_in,
            dim_h=dim_h,
            center_input=center_input,
            dataset_args=dataset_args
        )
        np.savez(outfile, **d)

    def save_images():
        w = model.W.get_value().T
        w = w.reshape((w.shape[0] // 10, 10, w.shape[1]))
        train.save_images(w, path.join(out_path, 'weights.png'))

        chain = f_chain()
        train.save_images(chain, path.join(out_path, 'chain.png'))

    # ========================================================================
    print_section('Getting gradients and building optimizer.')
    f_grad_shared, f_grad_updates, learning_args = set_optimizer(
        [X], cost, tparams, constants, updates, extra_outs, **learning_args)

    # ========================================================================
    print_section('Actually running (main loop)')
    monitor = SimpleMonitor()

    main_loop(
        train, valid, tparams,
        f_grad_shared, f_grad_updates, f_test, f_test_keys,
        save=save,
        save_images=save_images,
        monitor=monitor,
        out_path=out_path,
        name=name,
        extra_outs_keys=extra_outs_keys,
        **learning_args)

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    exp_dict = set_experiment(args)

    train(**exp_dict)