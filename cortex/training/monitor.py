'''
Module for monitor class.
'''

from collections import OrderedDict
from colorclass import Color
import cPickle as pkl
import numpy as np
import os
from terminaltables import AsciiTable

from ..utils.tools import update_dict_of_lists


class BasicMonitor(object):
    '''Simple monitor for displaying and saving results.

    Basic template monitor. Should be interchangeable in training for
    customized versions.

    Attributes:
        d: OrderedDict: dictionary of results.
        d_valid: OrderedDict: dictionary of results for validation.
    '''
    def __init__(self, modes=None):
        if modes is None:
            raise TypeError('Keyword value `modes` must be set.')

        self.stats = {}
        for mode in modes:
            if not isinstance(mode, str):
                raise TypeError('Modes must be strings.')
            d = OrderedDict()
            self.stats[mode] = d

        self.sections = {}

    def add_section(self, name, keys):
        self.sections[name] = keys

    def update(self, mode, **kwargs):
        update_dict_of_lists(self.stats[mode], **kwargs)

    def display(self):
        '''Displays the stats.

        '''
        for section in self.sections.keys():
            keys = []
            for k in self.stats.keys():
                keys.append(k)
                keys.append(u'\u0394' + k)
            table_data = [['Name'] + keys]
            for stat in sorted(self.sections[section]):
                stat_str = stat.replace('_grad_', u'\u03b4')
                stat_str = stat.replace('_delta_', u'\u0394')
                td = [stat_str]
                for mode in self.stats.keys():
                    if stat in self.stats[mode].keys():
                        s = self.stats[mode][stat][-1]
                        if len(self.stats[mode][stat]) > 1:
                            s_ = self.stats[mode][stat][-2]
                            ds = s - s_
                        else:
                            ds = float('inf')
                        if not isinstance(s, np.ndarray) and abs(s) < 1.:
                            td.append('%.2e' % s)
                            if ds > 0:
                                dss = Color('{autored}%.2e{/autored}' % ds)
                            else:
                                dss = Color('{autogreen}%.2e{/autogreen}' % ds)
                            td.append(dss)
                        elif not isinstance(s, np.ndarray):
                            td.append('%.2f' % s)
                            if ds > 0:
                                dss = Color('{autored}%.2f{/autored}' % ds)
                            else:
                                dss = Color('{autogreen}%.2f{/autogreen}' % ds)
                            td.append(dss)
                        else:
                            td.append(s)
                            td.append(None)
                    else:
                        td.append(None)
                        td.append(None)
                table_data.append(td)

            table = AsciiTable(table_data, section.title())
            #table.inner_row_border = True
            table.justify_columns[2] = 'right'
            print(table.table)

    def save(self, out_path):
        '''Saves a figure for the monitor

        Args:
            out_path: str
        '''

        plt.clf()
        np.set_printoptions(precision=4)
        font = {'size': 7}
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
