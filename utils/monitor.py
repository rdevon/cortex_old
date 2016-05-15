'''
Module for monitor class.
'''
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from collections import OrderedDict
import cPickle as pkl
import numpy as np
import os
import pprint
import signal
import time

from tools import check_bad_nums
from tools import update_dict_of_lists


class SimpleMonitor(object):
    '''Simple monitor for displaying and saving results.

    Basic template monitor. Should be interchangeable in training for
    customized versions.

    Attributes:
        d: OrderedDict: dictionary of results.
        d_valid: OrderedDict: dictionary of results for validation.
    '''
    def __init__(self, *args):
        self.d = OrderedDict()
        self.d_valid = OrderedDict()

    def update(self, **kwargs):
        update_dict_of_lists(self.d, **kwargs)

    def update_valid(self, **kwargs):
        update_dict_of_lists(self.d_valid, **kwargs)

    def add(self, **kwargs):
        for k, v in kwargs.iteritems():
            self.d[k] = v

    def simple_display(self, d):
        length = len('\t' ) + max(len(k) for k in d.keys())
        for k, vs in d.iteritems():
            s = '\t%s' % k
            s += ' ' * (length - len(s))
            s += ' |\t%.4f' % vs
            print s

    def display(self):
        '''Displays the stats.
        
        This uses some basic heuristics to get stats into rows with validation
        (if exists) as well as difference from last step.
        '''
        d = OrderedDict()
        for k in sorted(self.d):
            if not k.startswith('d_'):
                d[k] = [self.d[k][-1]]
                if k in self.d_valid.keys():
                    d[k].append(self.d_valid[k][-1])
                    if len(self.d_valid[k]) > 1:
                        d[k].append(self.d_valid[k][-1] - self.d_valid[k][-2])
                else:
                    d[k].append(None)

        length = len('\t' ) + max(len(k) for k in d.keys()) + len(' (train / valid) |')
        for k, vs in d.iteritems():
            s = '\t%s' % k
            if len(vs) > 1 and vs[1] is not None:
                s += ' (train / valid)'
            s += ' ' * (length - len(s))
            s += ' |\t%.4f' % vs[0]
            if len(vs) > 1 and vs[1] is not None:
                s += ' / %.4f  ' % vs[1]
            if len(vs) > 2:
                s += '\t' + unichr(0x394).encode('utf-8') + '=%.4f' % vs[2]
            print s

    def save(self, out_path):
        '''Saves a figure for the monitor
        
        Args:
            out_path: str
        '''
        
        plt.clf()
        np.set_printoptions(precision=4)
        font = {
            'size': 7
        }
        matplotlib.rc('font', **font)
        y = 2
        x = ((len(self.d) - 1) // y) + 1
        fig, axes = plt.subplots(y, x)
        fig.set_size_inches(20, 8)

        for j, (k, v) in enumerate(self.d.iteritems()):
            ax = axes[j // x, j % x]
            ax.plot(v, label=k)
            if k in self.d_valid.keys():
                ax.plot(self.d_valid[k], label=k + '(valid)')
            ax.set_title(k)
            ax.legend()

        plt.tight_layout()
        plt.savefig(out_path, facecolor=(1, 1, 1))
        plt.close()

    def save_stats(self, out_path):
        '''Saves the monitor dictionary.
        
        Args:
            out_path: str
        '''
        
        np.savez(out_path, **self.d)

    def save_stats_valid(self, out_path):
        '''Saves the valid monitor dictionary.
        
        Args:
            out_path: str
        '''
        np.savez(out_path, **self.d_valid)
