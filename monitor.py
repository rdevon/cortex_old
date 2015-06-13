'''
Module for monitor class.
'''

from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt


class Monitor(object):
    def __init__(self, tparams, first_order_stats=False, savefile='monitors.png'):
        self.tparams = tparams
        self.first_order_stats = first_order_stats
        self.params = [k for k in tparams]
        self.d = dict((k, {'mean': [], 'max': [], 'min': [], 'std': []})
                      for k in self.params)

        self.s = OrderedDict()
        self.savefile = savefile

    def update(self):
        # TODO: add grad norms and param norms here
        if self.first_order_stats:
            for p in self.params:
                p_mean = self.tparams[p].mean().eval()
                p_max = self.tparams[p].max().eval()
                p_min = self.tparams[p].min().eval()
                p_std = self.tparams[p].std().eval()
                self.d[p]['mean'].append(p_mean)
                self.d[p]['max'].append(p_max)
                self.d[p]['min'].append(p_min)
                self.d[p]['std'].append(p_std)

    def add_monitor(self, k, extra=False):
        if extra:
            self.d[k] = {'mean': [], 'max': [], 'min': [], 'std': []}
        else:
            self.s[k] = []

    def append_stats(self, **kwargs):
        for k, v in kwargs.iteritems():
            assert k in self.s.keys()
            self.s[k].append(v)

    def append_s_stat(self, s_mean, s_max, s_min, s_std, stat):
        assert stat in self.d.keys()
        self.d[stat]['mean'].append(s_mean)
        self.d[stat]['max'].append(s_max)
        self.d[stat]['min'].append(s_min)
        self.d[stat]['std'].append(s_std)

    def save(self):
        plt.clf()
        x = 2
        y = ((len(self.d) + len(self.s) - 1) // 2) + 1
        fig, axes = plt.subplots(y, x)
        fig.set_size_inches(18.5, 18.5)

        for j, (k, v) in enumerate(self.s.iteritems()):
            ax = axes[j // x, j % x]
            ax.plot(v, label=k)
            ax.set_title(k)
            ax.legend()

        j += 1

        for i, (k, v) in enumerate(self.d.iteritems()):
            ax = axes[(i+j) // x, (i+j) % x]
            for stat_k, stat_v in v.iteritems():
                if stat_k == 'std':
                    stat_mean = v['mean']
                    ax.fill_between(range(len(stat_mean)),
                                    [m - 2 * s for m, s in zip(stat_mean, stat_v)],
                                    [m + 2 * s for m, s in zip(stat_mean, stat_v)],
                                    alpha=0.3, facecolor='red')
                else:
                    ax.plot(stat_v, label=stat_k)
            ax.set_title(k)
            ax.legend()
        plt.tight_layout()
        plt.savefig(self.savefile)
        plt.close()

    def disp(self, epoch):
        s = 'Epoch %d | ' % epoch
        for k, v in self.s.iteritems():
            s += '%s: %.5f | ' % (k, v[-1].mean())
        print s