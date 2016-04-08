'''
Generic training scripts.
'''

import argparse
from collections import OrderedDict
from glob import glob
import numpy as np
import os
from os import path
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    RotatingMarker,
    SimpleProgress,
    Timer
)
import shutil
import theano
from theano import tensor as T
import time

import op
from tools import (
    check_bad_nums,
    itemlist,
    load_experiment,
    load_model,
    resolve_path,
    update_dict_of_lists
)


def make_argument_parser():
    '''Generic experiment parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    return parser

def set_experiment(args):
    '''Generic experiment setup method'''
    args = vars(args)
    load_model = args.pop('load_model')
    load_last = args.pop('load_last')
    args = OrderedDict((k, v) for k, v in args.iteritems() if v is not None)
    exp_dict = load_experiment(path.abspath(args['experiment']))
    exp_dict.update(args)

    if not 'out_path' in exp_dict.keys():
        exp_dict['out_path'] = resolve_path('$irvi_outs')

    exp_dict['out_path'] = path.join(exp_dict['out_path'], exp_dict['name'])

    out_path = exp_dict['out_path']
    print 'Saving to %s' % out_path
    if path.isfile(out_path):
        raise ValueError()
    elif not path.isdir(out_path):
        os.mkdir(path.abspath(out_path))

    experiment = exp_dict.pop('experiment')
    shutil.copy(path.abspath(experiment), path.abspath(out_path))

    if load_model is not None:
        model_to_load = load_model
    elif load_last:
        model_to_load = glob(path.join(out_path, '*last.npz'))
    else:
        model_to_load = None

    exp_dict['model_to_load'] = model_to_load
    return exp_dict

def set_model(create_model, model_to_load, unpack):
    '''Generic method for creating new or loading old model'''
    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack)
    else:
        models = create_model()
    return models

def set_params(tparams, updates):
    '''Sets params, removing updates from tparams'''
    all_params = OrderedDict((k, v) for k, v in tparams.iteritems())

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    print 'Learned model params: %s' % tparams.keys()
    print 'Saved params: %s' % all_params.keys()

    return tparams, all_params

def set_optimizer(inputs, cost, tparams, constants, updates, extra_outs,
                  optimizer=None, optimizer_args=None,
                  **learning_args):
    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=constants)

    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, inputs, cost, extra_ups=updates,
        extra_outs=extra_outs, **optimizer_args)

    return f_grad_shared, f_grad_updates, learning_args

def test(data_iter, f_test, f_test_keys, n_samples=None):
    '''Tests the model using a data iterator'''
    data_iter.reset()
    maxvalid = data_iter.n

    widgets = ['Validating (%s): ' % data_iter.mode, Percentage(), ' (', Timer(), ')']
    pbar    = ProgressBar(widgets=widgets, maxval=maxvalid).start()
    results = OrderedDict()
    while True:
        try:
            outs = data_iter.next()
            x = outs[data_iter.name]
            if n_samples is not None:
                x = x[:n_samples]
            r = f_test(x)
            results_i = dict((k, v) for k, v in zip(f_test_keys, r))
            update_dict_of_lists(results, **results_i)

            if data_iter.pos == -1:
                pbar.update(maxvalid)
            else:
                pbar.update(data_iter.pos)

        except StopIteration:
            print
            break

    for k, v in results.iteritems():
        results[k] = np.mean(v)

    data_iter.reset()

    return results

def validate(tparams, results, best_valid, e, best_epoch,
             save=None, valid_key=None, valid_sign=None, bestfile=None):
    '''Generic validation method'''
    valid_value = results[valid_key]
    if valid_sign == '-':
        valid_value *= -1

    if valid_value < best_valid:
        print 'Found best %s: %.2f' % (valid_key, valid_value)
        best_valid = valid_value
        best_epoch = e
        if save is not None and bestfile is not None:
            print 'Saving best to %s' % bestfile
            save(tparams, bestfile)
    else:
        print 'Best (%.2f) at epoch %d' % (best_valid, best_epoch)

    return best_valid, best_epoch

def main_loop(train, valid, tparams,
              f_grad_shared, f_grad_updates, f_test, f_test_keys,
              name=None,
              save=None,
              save_images=None,
              epochs=None,
              learning_rate=None,
              learning_rate_schedule=None,
              monitor=None,
              out_path=None,
              extra_outs_keys=None,
              output_every=None,
              **validation_args):
    '''Generic main loop'''

    best_valid = float('inf')
    best_epoch = 0

    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

    try:
        epoch_t0 = time.time()
        s = 0
        e = 0

        widgets = ['Epoch {epoch} (training {name}, '.format(epoch=e, name=name),
                   Timer(), '): ', Bar()]
        epoch_pbar = ProgressBar(widgets=widgets, maxval=train.n).start()
        training_time = 0
        while True:
            try:
                x = train.next()[train.name]
                if train.pos == -1:
                    epoch_pbar.update(train.n)
                else:
                    epoch_pbar.update(train.pos)

            except StopIteration:
                print
                epoch_t1 = time.time()
                dt_epoch = epoch_t1 - epoch_t0
                training_time += dt_epoch
                results = test(train, f_test, f_test_keys, n_samples=valid.n)
                results_valid = test(valid, f_test, f_test_keys)
                best_valid, best_epoch = validate(
                    tparams,
                    results_valid, best_valid, e, best_epoch,
                    bestfile=bestfile,
                    save=save, **validation_args)

                if monitor is not None:
                    monitor.update(**results)
                    monitor.update(dt_epoch=dt_epoch,
                                   training_time=training_time)
                    monitor.update_valid(**results_valid)
                    monitor.display()

                if save_images is not None:
                    save_images()

                e += 1
                epoch_t0 = time.time()

                if learning_rate_schedule is not None:
                    if 'decay' in learning_rate_schedule.keys():
                        learning_rate /= learning_rate_schedule['decay']
                        print 'Changing learning rate to %.5f' % learning_rate
                    elif e in learning_rate_schedule.keys():
                        lr = learning_rate_schedule[e]
                        print 'Changing learning rate to %.5f' % lr
                        learning_rate = lr

                widgets = ['Epoch {epoch} ({name}, '.format(epoch=e, name=name),
                           Timer(), '): ', Bar()]
                epoch_pbar = ProgressBar(widgets=widgets, maxval=train.n).start()

                continue

            if e > epochs:
                break

            rval = f_grad_shared(x)
            if output_every is not None and s % output_every == 0:
                #print rval
                if save_images is not None:
                    save_images()
            check_bad_nums(rval, extra_outs_keys)
            if check_bad_nums(rval[:1], extra_outs_keys[:1]):
                print 'Dying, found bad cost... Sorry (bleh)'
                exit()
            f_grad_updates(learning_rate)
            s += 1

    except KeyboardInterrupt:
        print 'Training interrupted'

    if out_path is not None:
        outfile = path.join(out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
        last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

        if save is not None:
            print 'Saving'
            save(tparams, outfile)
            save(tparams, last_outfile)
            print 'Done saving.'

    print 'Bye bye!'
