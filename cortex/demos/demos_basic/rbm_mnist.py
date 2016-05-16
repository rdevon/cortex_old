'''
Demo for training RBM with MNIST dataset.

Try with `python rbm_mnist.py rbm_mnist.yaml`.
'''

from collections import OrderedDict
import numpy as np
from os import path
import pprint
import theano
from theano import tensor as T

from cortex.datasets import load_data
from cortex.models.rbm import RBM, unpack
from cortex.utils import floatX
from cortex.utils.monitor import SimpleMonitor
from cortex.utils.preprocessor import Preprocessor
from cortex.utils.tools import get_trng, print_profile, print_section
from cortex.utils.training import (
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
    optimizer_args=None,
    learning_rate_schedule=None,
    batch_size=100,
    valid_batch_size=100,
    epochs=100,
    valid_key='nll',
    valid_sign='+',
    excludes=[]):
    if optimizer_args is None: optimizer_args = dict()
    return locals()

def init_inference_args(
    n_chains=10,
    persistent=False,
    n_steps=1):
    return locals()

def train(
    out_path=None, name='', model_to_load=None, save_images=True, test_every=None,
    dim_h=None, preprocessing=None,
    learning_args=None,
    inference_args=None,
    dataset_args=None):

    # ========================================================================
    if preprocessing is None: preprocessing = []
    if learning_args is None: learning_args = dict()
    if inference_args is None: inference_args = dict()
    if dataset_args is None: raise ValueError('Dataset args must be provided')

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

    preproc = Preprocessor(preprocessing)
    X_i = preproc(X, data_iter=train)
    inps = [X]

    # ========================================================================
    print_section('Loading model and forming graph')

    def create_model():
        model = RBM(dim_in, dim_h, v_dist=train.distributions[train.name],
                    mean_image=train.mean_image)
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
            np.zeros((inference_args['n_chains'], model.h_dist.dim)).astype(floatX),
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

    try:
        _, z_updates = model.update_partition_function(K=1000)
        f_update_partition = theano.function([], [], updates=z_updates)
    except NotImplementedError:
        f_update_partition = None

    H0 = model.trng.binomial(size=(10, model.h_dist.dim), dtype=floatX)
    s_outs, s_updates = model.sample(H0, n_steps=100)
    f_chain = theano.function(
        [], model.v_dist.get_center(s_outs['pvs']), updates=s_updates)

     # ========================================================================
    print_section('Setting final tparams and save function')
    excludes = learning_args.pop('excludes')
    tparams, all_params = set_params(tparams, updates, excludes=excludes)

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())
        d.update(
            dim_in=dim_in,
            dim_h=dim_h
        )
        np.savez(outfile, **d)

    def save_images():
        w = model.W.get_value().T
        w = w.reshape((10, w.shape[0] // 10, w.shape[1]))
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
        f_extra=f_update_partition,
        test_every=test_every,
        save=save,
        save_images=save_images,
        monitor=monitor,
        out_path=out_path,
        name=name,
        extra_outs_keys=extra_outs_keys,
        **learning_args)

if __name__ == '__main__':
    parser = make_argument_parser()
    parser.add_argument('-i', '--save_images', action='store_true')
    args = parser.parse_args()
    exp_dict = set_experiment(args)

    train(**exp_dict)