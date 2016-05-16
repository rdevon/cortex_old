'''
Demo for training VAE.

Try with `python vae.py vae_mnist.yaml`.
'''

from collections import OrderedDict
import numpy as np
from os import path
import pprint
import theano
from theano import tensor as T

from cortex.datasets import load_data
from cortex.models.helmholtz import Helmholtz, unpack
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
    learning_rate_schedule=None,
    l2_decay=0.,
    optimizer='rmsprop',
    optimizer_args=None,
    n_posterior_samples=20,
    reweight=False,
    batch_size=100,
    valid_batch_size=100,
    epochs=100,
    valid_key='-log p(x)',
    valid_sign='+',
    excludes=['gaussian_log_sigma', 'gaussian_mu']):
    if optimizer_args is None: optimizer_args = dict()
    return locals()

def train(
    out_path=None, name='', model_to_load=None, save_images=True, test_every=None,
    dim_h=None, rec_args=None, gen_args=None, prior='gaussian',
    preprocessing=None,
    learning_args=None,
    dataset_args=None):

    # ========================================================================
    if preprocessing is None: preprocessing = []
    if learning_args is None: learning_args = dict()
    if dataset_args is None: raise ValueError('Dataset args must be provided')
    learning_args = init_learning_args(**learning_args)

    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Learning args: %s' % pprint.pformat(learning_args)

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
        model = Helmholtz.factory(
            dim_h, train,
            prior=prior,
            rec_args=rec_args,
            gen_args=gen_args)

        models = OrderedDict()
        models[model.name] = model
        return models

    models = set_model(create_model, model_to_load, unpack)
    model = next((v for k, v in models.iteritems() if k in ['sbn', 'gbn', 'lbn']), None)
    posterior = model.posterior
    if not posterior.distribution.is_continuous:
        raise ValueError('Cannot perform VAE with posterior with distribution '
                         '%r' % type(posterior.distribution))
    tparams = model.set_tparams()
    print_profile(tparams)

    # ==========================================================================
    print_section('Getting cost')
    constants = []
    updates = theano.OrderedUpdates()
    n_posterior_samples = learning_args.pop('n_posterior_samples')
    reweight = learning_args.pop('reweight')
    results, samples, constants, updates = model(
        X_i, X, qk=None, pass_gradients=True,
        n_posterior_samples=n_posterior_samples, reweight=reweight)

    cost = results['cost']
    extra_outs = []
    extra_outs_keys = ['cost']

    l2_decay = learning_args.pop('l2_decay')
    if l2_decay is not False and l2_decay > 0.:
        print 'Adding %.5f L2 weight decay' % l2_decay
        l2_rval = model.l2_decay(l2_decay)
        cost += l2_rval.pop('cost')
        extra_outs += l2_rval.values()
        extra_outs_keys += l2_rval.keys()

    # ==========================================================================
    print_section('Test functions')
    f_test_keys = results.keys()
    f_test = theano.function([X], results.values())

    prior_samples, p_updates = model.sample_from_prior()
    f_prior = theano.function([], prior_samples, updates=p_updates)

    latent_vis = model.visualize_latents()
    f_latent = theano.function([], latent_vis)

    py = samples['py']
    f_py_h = theano.function([X], py)

    # ========================================================================
    print_section('Setting final tparams and save function')
    excludes = learning_args.pop('excludes')
    tparams, all_params = set_params(
        tparams, updates, excludes=excludes)

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())
        d.update(
            dim_h=dim_h,
            rec_args=rec_args,
            gen_args=gen_args
        )
        np.savez(outfile, **d)

    def save_images():
        p_samples = f_prior()
        p_samples = p_samples.reshape(
            (p_samples.shape[0] // 10, 10, p_samples.shape[1]))
        train.save_images(p_samples, path.join(out_path, 'prior_samples.png'))

        l_vis = f_latent()
        l_vis = l_vis.reshape((l_vis.shape[0] // 10, 10, l_vis.shape[1]))
        train.save_images(l_vis, path.join(out_path, 'latent_vis.png'))

        py_h = f_py_h(train.X[:100])
        train.save_images(py_h, path.join(out_path, 'py_h.png'))

    # ========================================================================
    print_section('Getting gradients and building optimizer.')
    f_grad_shared, f_grad_updates, learning_args = set_optimizer(
        inps, cost, tparams, constants, updates, extra_outs, **learning_args)

    # ========================================================================
    print_section('Actually running (main loop)')
    monitor = SimpleMonitor()

    main_loop(
        train, valid, tparams,
        f_grad_shared, f_grad_updates, f_test, f_test_keys,
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