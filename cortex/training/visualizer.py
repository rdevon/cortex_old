'''Visualizer class

'''

import theano
from theano import tensor as T


class Visualizer(object):

    def __init__(self, session, batch_size=None):
        from ..manager import get_manager
        self.session = session
        self.manager = get_manager()
        self.fs = []
        self.batch_size = batch_size

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

        tensor_idx = [i for i, a in enumerate(args)
                      if isinstance(a, T.TensorVariable)]
        nontensor_idx = [i for i in range(len(args)) if i not in tensor_idx]
        tensor_keys = [k for k, v in kwargs.iteritems()
                       if isinstance(v, T.TensorVariable)]
        nontensor_keys = [k for k in kwargs.keys() if k not in tensor_keys]

        f_viz = theano.function(
            self.session.inputs,
            [args[i] for i in tensor_idx] + [kwargs[k] for k in tensor_keys],
            updates=self.session.updates,
            on_unused_input='ignore')

        def viz(*inputs):
            ts = f_viz(*inputs)
            t_args = ts[:len(tensor_idx)]
            t_kwargs = dict(zip(tensor_keys, ts[len(tensor_idx):]))
            new_args = []
            t_i = 0
            for i in range(len(args)):
                if i in tensor_idx:
                    new_args.append(t_args[t_i])
                    t_i += 1
                else:
                    new_args.append(args[i])

            kwargs.update(**t_kwargs)
            return op(*new_args, **kwargs)

        self.fs.append(viz)

    def __call__(self, data_mode=None):
        self.session.reset_data(mode=data_mode)
        n = self.session.get_dataset_size(mode=data_mode)
        if self.batch_size is None:
            batch_size = n
        else:
            batch_size = self.batch_size
        inputs = self.session.next_batch(mode=data_mode, batch_size=batch_size)
        for f in self.fs:
            f(*inputs)