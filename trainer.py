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
             exclude_params=[], consider_constant=[], updates=[],
             weight_noise_amount=0.0):

    exclude_params = [tparams[ep] for ep in exclude_params]

    if inps is None:
        raise ValueError()
    if outs is None:
        raise ValueError()
    logger.info('Parameters are: %s' % tparams.keys())
    logger.info('The following params are excluded from learning: %s'
                % exclude_params)
    logger.info('The following variables are constant in learning gradient: %s'
                % consider_constant)

    #consider_constant = [tparams[cc] for cc in consider_constant]

    inps = inps.values()

    outs = flatten_dict(outs)

    out_keys = outs.keys()
    outs = outs.values()

    known_grads = costs.pop('known_grads')
    cost = costs.pop('cost')
    cost_keys = costs.keys()
    extra_costs = costs.values()

    # pop noise parameters since we dont need their grads
    keys_to_pop = [key for key in tparams.keys() if 'noise' in key]
    noise_params = dict([(key, tparams.pop(key)) for key in keys_to_pop])

    # add noise here and pass to optimizers extra_ups
    trng = RandomStreams(np.random.randint(int(1e6)))
    noise_updates = [(p, trng.normal(p.get_value().shape, avg=0,
                                     std=weight_noise_amount, dtype=p.dtype))
                     for p in noise_params.values()]
    updates.update(noise_updates)

    tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
        if v not in exclude_params)
    grads = T.grad(cost, wrt=itemlist(tparams), known_grads=known_grads,
                   consider_constant=consider_constant)

    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, inps, cost,
        extra_ups=updates,
        extra_outs=outs + extra_costs,
        exclude_params=exclude_params)

    return f_grad_shared, f_grad_updates, ['cost'] + out_keys + cost_keys


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
    display_interval = hyperparams['display_interval']

    logger.info('Fetching model')
    model = experiment.get_model()
    data = model.pop('data')

    logger.info('Calculating costs')
    costs = experiment.get_costs(**model)

    logger.info('Getting gradient functions')
    f_grad_shared, f_update, keys = get_grad(optimizer, costs, **model)

    logger.info('Initializing monitors')
    monitor = Monitor(model['tparams'])
    for k in keys:
        if 'cost' in k or 'energy' in k or 'reward' in k:
            monitor.add_monitor(k)

    try:
        logger.info('Training')
        for e in xrange(epochs):
            ud_start = time.time()

            s = 0
            i = 0
            while True:
                try:
                    inps = data['train'].next()
                except StopIteration:
                    monitor.disp(e)
                    break

                rvals = f_grad_shared(*inps)
                rval_dict = dict((k, r) for k, r in zip(keys, rvals))
                monitor.append_stats(**{k: v for k, v in rval_dict.iteritems()
                                        if 'cost' in k or 'energy' in k or 'reward' in k})
                if display_interval is not None and s == display_interval:
                    s = 0
                    monitor.disp(e)
                    if out_path is not None:
                        rnn_samples = rval_dict['cond_gen_gru_x'][:, :10]
                        data['train'].dataset.save_images(
                            rnn_samples,
                            path.join(out_path, 'rnn_samples_%d.png' % (i % 20)))
                        rbm_samples = rval_dict['rbm_x']
                        data['train'].dataset.save_images(
                            rbm_samples,
                            path.join(out_path, 'rbm_samples_%d.png' % (i % 20)))
                        i += 1
                else:
                    s += 1

                check_bad_nums(rval_dict, data['train'].count)

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