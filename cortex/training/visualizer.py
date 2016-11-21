'''Visualizer class

'''

from os import path
from progressbar import Bar, ProgressBar, Percentage, Timer
import theano
from theano import tensor as T

from ..utils.logger import get_class_logger


class Visualizer(object):

    def __init__(self, session, batch_size=None):
        from ..manager import get_manager
        self.session = session
        self.manager = get_manager()
        self.fs = []
        self.batch_size = batch_size
        self.logger = get_class_logger(self)
        self.f_names = []

    def add(self, op, *args, **kwargs):
        from ..manager import resolve_tensor_arg

        if isinstance(op, str):
            data_name, op, _ = resolve_tensor_arg(op)
            data = self.manager.datasets[data_name]['train']
            op = getattr(data, op)
        elif hasattr(op, __call__):
            data = None
        else:
            raise TypeError

        args, kwargs = self.session.resolve_op_args(args, kwargs)

        tensors = []

        def process_list(l):
            l_ = []
            for a in l:
                if isinstance(a, (T.TensorVariable,
                                  T.sharedvar.SharedVariable)):
                    l_.append('&tensor_{}'.format(len(tensors)))
                    tensors.append(a)
                elif isinstance(a, list):
                    l_.append(process_list(a))
                elif isinstance(a, dict):
                    keys = a.keys()
                    values = a.values()
                    values = process_list(values)
                    l_.append(dict((k, v) for k, v in zip(keys, values)))
                else:
                    l_.append(a)
            return l_
        
        def unpack_list(l, tensors):
            l_ = []
            for a in l:
                if isinstance(a, str) and '&tensor' in a:
                    i = int(a[8:])
                    l_.append(tensors[i])
                elif isinstance(a, list):
                    l_.append(unpack_list(a, tensors))
                elif isinstance(a, dict):
                    keys = a.keys()
                    values = a.values()
                    values = unpack_list(values, tensors)
                    l_.append(dict((k, v) for k, v in zip(keys, values)))
                else:
                    l_.append(a)
            return l_
        
        args_ = process_list(args)
        kwargs_ = dict((k, v) for k, v
            in zip(kwargs.keys(), process_list(kwargs.values())))

        f_viz = theano.function(
            self.session.inputs, tensors, updates=self.session.updates,
            on_unused_input='ignore')

        def viz(*inputs, **extra_args):
            ts = f_viz(*inputs)
            args = unpack_list(args_, ts)
            values = unpack_list(kwargs_.values(), ts)
            kwargs = dict((k, v) for k, v in zip(kwargs_.keys(), values))
            kwargs.update(**extra_args)
            
            if 'name' in kwargs.keys() and self.manager.out_path is not None:
                name = kwargs.pop('name')
                kwargs['out_file'] = path.join(self.manager.out_path, name + '.png')
            return op(*args, **kwargs)

        self.fs.append(viz)
        self.f_names.append(kwargs.get('name', 'Viz'))
        
    def run(self, idx, inputs=None, data_mode=None, **extra_args):
        self.session.reset_data(mode=data_mode)
        n = self.session.get_dataset_size(mode=data_mode)
        inputs = self.session.next_batch(mode=data_mode, batch_size=n)
            
        try:
            return self.fs[idx](*inputs, **extra_args)
        except IndexError:
            raise IndexError('Visualization function index {} does not '
                             'exist'.format(len(self.fs)))

    def __call__(self, data_mode=None, inputs=None):
        widgets = ['Visualizing (please wait): ', Bar()]
        if inputs is None:
            self.session.reset_data(mode=data_mode)
            n = self.session.get_dataset_size(mode=data_mode)
            if self.batch_size is None:
                batch_size = n
            else:
                batch_size = self.batch_size
            inputs = self.session.next_batch(mode=data_mode, batch_size=batch_size)
        pbar = ProgressBar(widgets=widgets, maxval=len(self.fs)).start()
        for i, f in enumerate(self.fs):
            self.logger.debug('Visualizer function `%s`' % self.f_names[i])
            f(*inputs)
            pbar.update(i)
        pbar.update(i + 1)
        print
