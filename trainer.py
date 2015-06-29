'''
Main training module.
'''

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt

import argparse
from collections import OrderedDict
import gru
import importlib
from monitor import Monitor
import numpy as np
import op
import os
from os import path
import time
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T
from tools import check_bad_nums
from tools import itemlist
from tools import flatten_dict

import logging
try:
    import logger
    logger = logger.setup_custom_logger('nmt', logging.ERROR)
except ImportError:
    logger = logging.getLogger(__name__)


def get_grad(optimizer, costs, tparams, inps=None, outs=None,
             vouts=None, exclude_params=[], consider_constant=[], updates=[],
             weight_noise_amount=0., **kwargs):

    if inps is None:
        raise ValueError()

    exclude_params = [tparams[ep] for ep in exclude_params]

    logger.info('Parameters are: %s' % tparams.keys())
    logger.info('The following params are excluded from learning: %s'
                % exclude_params)
    logger.info('The following variables are constant in learning gradient: %s'
                % consider_constant)

    known_grads = costs.pop('known_grads')
    cost = costs.pop('cost')

    inps = inps.values()
    if vouts is None and outs is not None:
        extra_outs = flatten_dict(costs).values() + flatten_dict(outs).values()
    else:
        extra_outs = []

    # pop noise parameters since we dont need their grads
    _tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if v not in updates.keys())
    keys_to_pop = [key for key in _tparams.keys() if 'noise' in key]
    noise_params = dict([(key, _tparams.pop(key)) for key in keys_to_pop])

    # add noise here and pass to optimizers extra_ups
    #trng = RandomStreams(np.random.randint(int(1e6)))
    trng = RandomStreams(505)
    noise_updates = [(p, trng.normal(p.shape, avg=0.,
                                     std=weight_noise_amount, dtype=p.dtype))
                     for p in noise_params.values()]
    updates.update(noise_updates)
    grads = T.grad(cost, wrt=itemlist(_tparams), known_grads=known_grads,
                   consider_constant=consider_constant)

    lr = T.scalar(name='lr')

    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, _tparams, grads, inps, cost, extra_ups=updates,
        extra_outs=extra_outs, exclude_params=exclude_params)

    return f_grad_shared, f_grad_updates

def make_fn(inps, d, updates=None):
    if updates is None:
        updates = theano.OrderedUpdates()
    d = flatten_dict(d)
    keys = d.keys()
    values = d.values()
    fn = theano.function(inps.values(), values, on_unused_input='warn',
                         updates=updates)
    return lambda *inps: dict((k, v) for k, v in zip(keys, fn(*inps)))

def make_fn_given_fgrad(keys, start_idx):
    return lambda *outs: dict((k, v)
        for k, v in zip(keys, outs[start_idx: start_idx + len(keys)]))

def train(experiment_file, out_path=None, **kwargs):
    if experiment_file.split('.')[-1] == 'py':
        experiment_file = experiment_file[:-3]

    logger.info('Loading experiment %s' % experiment_file)
    experiment = importlib.import_module(experiment_file)
    hyperparams = experiment.default_hyperparams
    for k in kwargs.keys():
        assert k in hyperparams.keys()
    hyperparams.update(**kwargs)

    lrate = hyperparams['learning_rate']
    optimizer = hyperparams['optimizer']
    epochs = hyperparams['epochs']
    display_interval = hyperparams['disp_freq']

    logger.info('Fetching model')
    model = experiment.get_model(**hyperparams)
    data = model.pop('data')

    logger.info('Calculating costs')
    costs = experiment.get_costs(**model)

    logger.info('Getting gradient functions')
    f_grad_shared, f_update = get_grad(optimizer, costs, **model)

    logger.info('Getting validation functions')
    if model.get('vouts', False) and model['vouts'] is not None:
        v_costs = experiment.get_costs(inps=model['inps'], outs=model['vouts'])
        v_costs.pop('known_grads')
        cost_fn = make_fn(model['inps'], v_costs, updates=model['vupdates'])
        err_fn = make_fn(model['inps'], model['errs'])
        out_fn = make_fn(model['inps'], model['vouts'])
        valid_graph = True
    else:
        cost_fn = make_fn_given_fgrad(['cost'] + flatten_dict(costs).keys(), 0)
        err_fn = make_fn_given_fgrad([], 0)
        out_fn = make_fn_given_fgrad(flatten_dict(model['outs']).keys(),
                                     len(costs) + 1)

        valid_graph = False

    logger.info('Initializing monitors')
    monitor = Monitor(model['tparams'], data, cost_fn, err_fn, out_fn,
                      early_stopping=False, hyperparams=hyperparams)

    try:
        logger.info('Training')
        for e in xrange(epochs):
            ud_start = time.time()

            s = 0
            i = 0
            while True:
                atime = time.time()
                try:
                    inps = data['train'].next()
                except StopIteration:
                    monitor.disp(e, data['train'].count)
                    break

                outs = f_grad_shared(*inps)
                if valid_graph:
                    train_c, train_e, train_o = monitor.update(*inps)
                else:
                    train_c, train_e, train_o = monitor.update(*outs)

                btime = time.time()
                if display_interval is not None and s == display_interval:
                    s = 0
                    monitor.disp(e, data['train'].count, btime - atime)
                    if out_path is not None:
                        save_images = data['train'].dataset.save_images
                        rnn_samples = train_o['cond_gen_gru_x'][:, :10]
                        rnn_probs = train_o['cond_gen_gru_p'][:, :10]
                        save_images(
                            rnn_samples,
                            path.join(out_path, 'rnn_samples_%d.png' % (i % 20)))
                        save_images(
                            rnn_probs,
                            path.join(out_path, 'rnn_probs_%d.png' % (i % 20)))

                        rbm_samples = train_o['rbm_x']
                        rbm_probs = train_o['rbm_p']
                        save_images(
                            rbm_samples,
                            path.join(out_path, 'rbm_samples_%d.png' % (i % 20)))
                        save_images(
                            rbm_probs,
                            path.join(out_path, 'rbm_probs_%d.png' % (i % 20)))
                        i += 1
                else:
                    s += 1

                #check_bad_nums(rval_dict, data['train'].count)

                f_update(lrate)

            ud = time.time() - ud_start

    except KeyboardInterrupt:
        logger.info('Training interrupted')


def make_argument_parser():
    '''
    Arg parser for simple runner.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', help='Experiment module')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--hyperparams', default=None,
                        help=('Comma separated list of '
                              '<key>:<value> pairs'))
    parser.add_argument('-o', '--out_dir', default=None,
                        help='output directory for files')

    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.info('Verbose logging')
    if args.out_dir is not None:
        if path.isfile(args.out_dir):
            raise ValueError('Out path is a file')
        if not path.isdir(args.out_dir):
            os.mkdir(args.out_dir)

    train(args.experiment, out_path=args.out_dir)