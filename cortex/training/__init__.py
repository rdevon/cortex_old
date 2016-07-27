'''Trainer class.

Does typical training work.

'''

from collections import OrderedDict
import logging
import numpy as np
import theano
from theano import tensor as T


logger = logging.getLogger('cortex')


def train(module, cost, tparams, updates, constants, f_test=None, f_save=None,
          f_viz=None, f_outs=None, f_profile=None, test_every=10, show_every=10,
          monitor_gradients=False):
    print_section('Getting gradients and building optimizer.')

    excludes = module.learning_args.pop('excludes', [])
    tparams, all_params = set_params(tparams, updates, excludes=excludes)
    f_grad_shared, f_grad_updates, learning_args = set_optimizer(
        module.inputs.values(), cost, tparams, constants, updates, [],
        **module.learning_args)

    monitor = SimpleMonitor()


def main_loop(train, valid,
              f_grad_shared, f_grad_updates, f_test,
              f_test_keys=None,
              input_keys=None,
              f_extra=None, f_outs=None, f_profile=None,
              test_every=None, show_every=None, output_every=None,
              monitor_gradients=False,
              name=None,
              save=None, save_images=None,
              epochs=None, learning_rate=None, learning_rate_scheduler=None,
              monitor=None,):

    best_valid = float('inf')
    best_epoch = 0

    if learning_rate_scheduler is not None:
        learning_rate_scheduler = Scheduler(**learning_rate_scheduler)
        learning_rate = [v['learning_rate'] for v in learning_rate_scheduler.d.values()]
    elif isinstance(learning_rate, float):
        learning_rate = (learning_rate,)

    if input_keys is None:
        input_keys = [train.name]

    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))
    else:
        bestfile = None

    try:

            except StopIteration:
                if (test_every is None) or ((e + 1) % test_every == 0):
                    print

                    if f_extra is not None:
                        logging.info('Performing initial evaluation function...')
                        f_extra()

                    epoch_t1 = time.time()
                    dt_epoch = epoch_t1 - epoch_t0
                    training_time += dt_epoch
                    results = test(train, f_test, f_test_keys, input_keys)
                    results_valid = test(valid, f_test, f_test_keys, input_keys)

                    best_valid, best_epoch = validate(
                        results_valid, best_valid, e, best_epoch,
                        bestfile=bestfile,
                        save=save, **validation_args)

                    if monitor is not None:
                        monitor.update(**results)
                        if monitor_gradients:
                            rval.pop('cost', None)
                            monitor.update(**rval)
                        monitor.update(dt_epoch=dt_epoch,
                                       training_time=training_time,
                                       learning_rate=learning_rate[0])
                        monitor.update_valid(**results_valid)
                        monitor.display()

                        if out_path is not None:
                            monitor.save(path.join(out_path, 'monitor.png'))
                            monitor.save_stats(
                                path.join(out_path, 'stats_train.npz'))
                            monitor.save_stats_valid(
                                path.join(out_path, 'stats_valid.npz'))

                if (save_images is not None
                    and out_path is not None
                    and (show_every is None or ((e + 1) % show_every == 0))):
                    print('Saving images...')
                    save_images()

                e += 1

                if learning_rate_scheduler is not None:
                    learning_rate = learning_rate_scheduler(e)

                epoch_t0 = time.time()
                widgets = ['Epoch {epoch} ({name}, '.format(epoch=e, name=name),
                           Timer(), '): ', Bar()]

                continue

            if e > epochs:
                break



    except KeyboardInterrupt:
        print 'Training interrupted.'
    except:
        logger.exception('Exception reached during training')
        raise

    try:
        if out_path is not None:
            outfile = path.join(out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
            last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

            if save is not None:
                logging.info('Saving')
                save(outfile)
                save(last_outfile)
                logging.info('Done saving.')
    except KeyboardInterrupt:
        print 'Saving interupted.'

    print 'Main loop finished.'


class Trainer(object):
    '''Trainer class for cortex.

    Trainer processes and organizes scripts for use in cortex.

    '''

    def __init__(self)
        epoch_t0 = time.time()
        training_time = 0
        self.generator = _generator()
        self.()

    def reset(self):
        self.epoch += 1
        if epoch >= self.epochs:
            raise StopIteration
        widgets = ['Epoch {epoch} (training {name}, '.format(
            epoch=epoch, name=name), Timer(), '): ', Bar()]
        self.epoch_pbar = ProgressBar(widgets=widgets, maxval=train.n).start()
        train.reset()

    def next(self):
        while True:
            try:
                inputs = train.next()
            except StopIteration:
                self.reset()

            self.epoch_pbar.update(train.pos)
            rval = self.f_grad_shared(*inputs)

            if check_bad_nums(rval):
                check_bad_nums(f_test(*inputs))
                if f_outs is not None:
                    check_bad_nums(f_outs(*inps))
                raise RuntimeError('Dying, found bad cost... Sorry (bleh)')

            self.f_grad_updates(*learning_rate)

    def set_optimizer(self, optimizer='sgd', optimizer_args=None):
        '''Sets the parameter update functions with optimizer.

        Args:
            inputs (T.tensor): input variables.
            cost (T.scalar): cost
            tparams (OrderedDict): directionary of tensor parameters
            constants (list): list of constant tensors.
            updates (theano.OrderedUpdates): updates.
            extra_outs (list): list of extra output tensors.
            optimizer (Optional[str]): optimizer string. See `utils.op` for details.
                Defaults to `sgd`.
            optimizer_args (Optional[dict]): optional arguments for optimizer.
            **learning_args: extra kwargs for learning not used.

        Returns:
            theano.function: gradient function.
            theano.function: update function.
            dict: extra learning keyword arguments.

        '''
        session = self.session
        manager = self.manager

        cost = sum(session.costs)
        tparams = manager.tparams

        if optimizer_args is None:
            optimizer_args = dict()
        grads = T.grad(
            cost, wrt=tparams.values(), consider_constant=session.constants)
        grads = OrderedDict((k, g) for k, g in zip(tparams.keys(), grads))

        lr = T.scalar(name='lr')
        f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
            lr, tparams, grads, session.inputs, cost, extra_ups=session.updates,
            **optimizer_args)

        self.f_grad_shared = f_grad_shared
        self.f_grad_updates = f_grad_updates

class Evaluator(object):
    def __init__(self, session, valid_stat='cost'):
        self.session = session
        self.f_stats = theano.function(session.inputs, self.session.stats)

    def next(self, mode=None):
        widgets = ['Testing (%s set): ' % mode, Percentage(),
                   ' (', Timer(), ')']
        pbar    = ProgressBar(widgets=widgets, maxval=maxvalid).start()
        results = OrderedDict()
        self.session.reset_data(mode=mode)

        while True:
            try:
                outs = data_iter.next()
                inps = [outs[k] for k in input_keys]
                r = self.f_stats()

                for k, v in r.iteritems():
                    if isinstance(v, theano.sandbox.cuda.CudaNdarray):
                        r[k] = np.asarray(v)
                update_dict_of_lists(results, **r)

                if data_iter.pos == -1:
                    pbar.update(maxvalid)
                else:
                    pbar.update(data_iter.pos)

            except StopIteration:
                print
                break

        for k, v in results.iteritems():
            try:
                results[k] = np.mean(v)
            except Exception as e:
                logging.error(k)
                logging.error(v)
                raise e


class Visualizer(object):
    pass


def validate(results, best_valid, e, best_epoch, save=None, valid_key=None,
             valid_sign=None, bestfile=None, **kwargs):
    '''Generic validation method.

    Compares the validation result against previous best.

    Args:
        results (OrderedDict): dictionary of np.array results.
        best_valid (float): Best pervious value.
        e (int): Epoch
        best_epoch (int): Epoch for best_valid.
        save (function): Method for saving params.
        valid_key (str): Key from results to test against best_valid.
        bestfile (str): Path to best file.

    Returns:
        float: best valid
        int: best epoch

    '''
    warn_kwargs(None, kwargs)

    valid_value = results[valid_key]
    if valid_sign == '-':
        valid_value *= -1

    if valid_value < best_valid:
        print 'Found best %s: %.2f' % (valid_key, valid_value)
        best_valid = valid_value
        best_epoch = e
        if save is not None and bestfile is not None:
            print 'Saving best to %s' % bestfile
            save(bestfile)
    else:
        print 'Best (%.2f) at epoch %d' % (best_valid, best_epoch)

    return best_valid, best_epoch

def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_argument_parser_trainer()
    args = parser.parse_args()
    exp_dict = set_experiment(args)
    module = ModuleContainer(
        path.abspath(exp_dict.pop('module')),
        preprocessing=exp_dict.pop('preprocessing', None),
        name=exp_dict.pop('name', None),
        out_path=exp_dict.pop('out_path', None))

    module.update(exp_dict)
    show_every = exp_dict.pop('show_every', 10)
    test_every = exp_dict.pop('test_every', 10)
    monitor_gradients = exp_dict.pop('monitor_gradients', False)
    model_to_load = exp_dict.pop('model_to_load', None)

    trainer = Trainer()
    trainer.run(module, show_every=show_every, test_every=test_every,
                model_to_load=model_to_load, monitor_gradients=monitor_gradients)