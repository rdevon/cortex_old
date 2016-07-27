'''Module for inspector.

'''


import numpy as np

from . import set_eval_functions
from .container import load_module


class Inspector(object):
    def __init__(self, module, model_to_load=None, strict=True):
        if isinstance(module, str):
            module = load_module(module, strict=strict)
        self.module = module
        self.set()

    def set(self, **kwargs):
        eval_methods = set_eval_functions(self.module, **kwargs)
        self.eval_keys = []
        for k, v in eval_methods.iteritems():
            self.eval_keys.append(k)
            setattr(self, k, v)

    def show(self):
        for k in self.eval_keys:
            self.__dict__[k]()

    def get_stat(self, stat, mode='train'):
        if not hasattr(self, 'f_stats'):
            raise NotImplementedError('f_stats was not loaded.')

        stats = self.f_stats(mode=mode)
        if not stat in stats.keys():
            raise KeyError('Stat %s not found: (%s)' % (stat, stats.keys()))
        return stats[stat]

    def save_stat(self, stat, out_file, mode='train'):
        stat = self.get_stat(stat, mode=mode)
        np.save(out_file, stat)