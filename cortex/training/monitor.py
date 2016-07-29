'''
Module for monitor class.
'''

from collections import OrderedDict
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
            table_data = [['Name'] + self.stats.keys()]
            for stat in self.sections[section]:
                td = [stat]
                for mode in self.stats.keys():
                    td.append(self.stats[mode][stat][-1])
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
