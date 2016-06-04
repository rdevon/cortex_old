'''Trainer class.

Does typical training work.

'''

from collections import OrderedDict
import logging
import theano
from theano import tensor as T

from . import floatX, intX
from .preprocessor import Preprocessor
from .tools import get_trng, print_section
from .training import set_model, main_loop
from ..datasets import dataset_factory


class Trainer(object):

    default_learning_args = dict()

    def __init__(self, name='trainer', dataset_args=None, learning_args=None,
                 preprocessing=None, out_path=None):
        if learning_args is None: learning_args = dict()
        if dataset_args is None:
            raise ValueError('`dataset_args` not provided.')

        self.logger = logging.getLogger('cortex')
        self.name = name
        self.trng = get_trng()
        self.preprocessor = Preprocessor(preprocessing)
        self.learning_args = self.initialize_args(
            self.default_learning_args, **learning_args)
        self.dataset_args = dataset_args
        self.out_path = out_path

    def initialize_args(self, defaults, **kwargs):
        extra_keys = [k for k in kwargs.keys() if k not in defaults.keys()]
        if len(extra_keys) > 0:
            raise ValueError('Unknown args provided: %s' % extra_keys)

        args = defaults.copy()
        args.update(**kwargs)
        return args

    def setup_data(self, resolve_dataset):
        print_section('Setting up data')
        data = dataset_factory(resolve_dataset, **self.dataset_args)
        return data

    def make_inputs(self, data):
        print_section('Setting inputs')

        d = data.next()
        self.inps = OrderedDict()
        for k, v in d.iteritems():
            if v.ndim == 1:
                C = T.vector
            elif v.ndim == 2:
                C = T.matrix
            elif v.ndim == 3:
                C = T.tensor3
            else:
                raise ValueError('Data dim over 3 not supported.')

            if v.dtype == floatX:
                dtype = floatX
            elif v.dtype == intX:
                dtype = intX
            else:
                raise ValueError('dtype %s not supported' % v.dtype)

            X = T.tensor3(k, dtype=dtype)
            self.inps[k] = (X, data.distributions[k])
        self.logger.debug('Model has the following inputs: %s' % self.inps)
        data.reset()

    def create_model(self):
        raise NotImplementedError()

    def unpack(self, **kwargs):
        raise NotImplementedError()

    def form_model(self, model_to_load=None):
        self.models = set_model(self.create_model, model_to_load,
                                self.unpack)

        self.tparams = OrderedDict()
        for k, v in self.models.iteritems():
            self.tparams.update(**v.set_tparams())

    def set_cost(self, models):
        raise NotImplementedError()

    def set_f_test(self, results):
        raise NotImplementedError()

    def go(self, cost, updates, constants, data, f_test=None, test_every=None,
           show_every=None):
        print_section('Getting gradients and building optimizer.')

        f_grad_shared, f_grad_updates, learning_args = set_optimizer(
            self.inps, cost, self.tparams, constants, updates, [],
            **learning_args)

        print_section('Actually running (main loop)')
        monitor = SimpleMonitor()

        main_loop(
            data['train'], data['valid'], tparams,
            f_grad_shared, f_grad_updates, f_test,
            save=self.save, save_images=self.save_images,
            monitor=monitor,
            out_path=self.out_path,
            name=self.name,
            test_every=test_every,
            show_every=show_every,
            **learning_args)

    def save(self, outfile, **kwargs):
        d = dict((k, v.get_value()) for k, v in self.tparams.items())
        d.update(dataset_args=self.dataset_args, **kwargs)
        np.savez(outfile, **d)


def train(C, model_to_load=None, test_every=None, show_every=None,
          resolve_dataset=None, **kwargs):
    '''Train using a trainer.

    '''
    trainer = C(**kwargs)
    data = trainer.setup_data(resolve_dataset)
    trainer.make_inputs(data['train'])
    models, tparams = trainer.form_model()
    exit()
    print_profile(tparams)
    results = trainer.set_cost(models)
    f_test = trainer.set_f_test(results)
    trainer.go(results['cost'], data, f_test=f_test, test_every=test_every,
               show_every=show_every, out_path=out_path)