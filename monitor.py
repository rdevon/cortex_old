'''
Module for monitor class.
'''
import matplotlib
from matplotlib import pylab as plt
import numpy as np
import cPickle as pkl
import os
import time
import signal

from collections import OrderedDict
from tools import check_bad_nums
matplotlib.use('Agg')
from matplotlib import pylab as plt


class SimpleMonitor(object):
    def __init__(self, *args):
        self.d = OrderedDict()

    def update(self, **kwargs):
        for k, v in kwargs.iteritems():
            if not self.d.get(k, False):
                self.d[k] = [v]
            else:
                self.d[k].append(v)

    def display(self, e):
        s = 'Samples: %d' % e
        for k, v in self.d.iteritems():
            try:
                s += ' | %s: %.2f' % (k, v[-1])
            except TypeError as e:
                print 'Error: %s, %s' % (k, v[-1])
                raise e
        print s

    def save(self, out_path):
        plt.clf()
        y = 2
        x = ((len(self.d) - 1) // y) + 1
        fig, axes = plt.subplots(y, x)
        fig.set_size_inches(20, 5)
        fig.patch.set_alpha(.1)

        for j, (k, v) in enumerate(self.d.iteritems()):
            ax = axes[j // x, j % x]
            ax.plot(v, label=k)
            ax.set_title(k)
            ax.legend()
            ax.patch.set_alpha(0.5)

        plt.tight_layout()
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', transparent=True)
        plt.close()


class Monitor(object):
    """Training monitor.

    Early stopping is done for each err_keys variable on validation set
    separately and following files are saved:
        - model_<err_key>_<timestamp>.npz : model parameters
        - config_<timestamp>.pkl : model hyper parameters
        - timing_<timestamp>.pkl : monitoring channel timings
    """
    def __init__(self, tparams, data, cost_fn, err_fn, out_fn, name='model',
                 sample_fn=None, first_order_stats=False, savefile='monitors.png',
                 early_stopping=False, hyperparams=None):

        self.__dict__.update(locals())
        print self.sample_fn
        del self.self

        self.timestamp = int(time.time())

        self.param_keys = [k for k in tparams]
        self.params = dict((k, {'mean': [], 'max': [], 'min': [], 'std': []})
                           for k in self.param_keys)

        self.stats = OrderedDict(train=OrderedDict())
        self.samples = OrderedDict(train=OrderedDict())
        if data['valid'] is not None and data['valid'].dataset is not None:
            self.stats['valid'] = OrderedDict()
            self.samples['valid'] = OrderedDict()
        if data['test'] is not None and data['test'].dataset is not None:
            self.stats['test'] = OrderedDict()
            self.samples['test'] = OrderedDict()
        self.err_fn = err_fn

        if self.early_stopping:
            raise NotImplementedError('Need to fix this!')
            assert self.hyperparams is not None, "Specify hyper parameters!"
            assert 'valid' in data and data['valid'].dataset is not None,\
                "Validation set is not provided!"
            self.best_models = [OrderedDict() for _ in self.err_keys]
            self.best_errs = [np.inf for _ in self.err_keys]

    def append_stats(self, dataset, stats):
        for k, v in stats.iteritems():
            if k in self.stats[dataset]:
                self.stats[dataset][k].append(v)
            else:
                self.stats[dataset][k] = [v]

    def get_stats(self, *inps):
        return self.cost_fn(*inps), self.err_fn(*inps), self.out_fn(*inps)

    def update(self, *inps):
        train_c, train_e, train_o = self.get_stats(*inps)
        #r_hat = train_o['logistic_y_hat']
        #a_max = np.argmax(r_hat)
        #target_value = r_hat[a_max]
        #max_v = a_max
        #for i in range(a_max,len(r_hat)):
        #    if(target_value==r_hat[i]):
        #        max_v = i
        #train_tweets = self.data['train'].next_tweet
        #train_tweets = train_tweets[a_max:max_v][:]
        #self.data['train'].detokenize(train_tweets)

        check_bad_nums(dict((k, v) for k, v in train_c.iteritems()),
                       self.data['train'].count)

        check_bad_nums(dict((k, v) for k, v in train_o.iteritems()),
                       self.data['train'].count)
        self.append_stats('train', train_c)
        self.append_stats('train', train_e)
        #self.append_stats('train', dict(sm_out=train_o['softmax_y_hat'].mean()))
        #print 'train'
        #print inps[1]
        #print train_o['softmax_y_hat']

        if self.sample_fn is not None:
            self.samples['train'] = self.sample_fn(*inps)

        if self.data['valid'] is not None and self.data['valid'].dataset is not None:
            try:
                inps = self.data['valid'].next()
            except StopIteration:
                return
            valid_c, valid_e, valid_o = self.get_stats(*inps)
            #valid_costs, valid_outs, valid_errs =\
            #    self._validate(self.data['valid'])
            self.append_stats('valid', valid_c)
            self.append_stats('valid', valid_e)
            #self.append_stats('valid', dict(sm_out=valid_o['softmax_y_hat'].mean()))
            #print 'valid'
            #print inps[1]
            #print valid_o['softmax_y_hat']

            # Early stopping mechanism
            if self.early_stopping:
                raise NotImplementedError('Need to fix!')
                self._track_current_model(valid_errs)

            if self.sample_fn is not None:
                self.samples['valid'] = self.sample_fn(*inps)

        if self.data['test'] is not None and self.data['test'].dataset is not None:
            try:
                inps = self.data['test'].next()
            except StopIteration:
                return
            test_c, test_e, test_o = self.get_stats(*inps)
            #test_costs, test_outs, test_errs =\
            #    self._validate(self.data['test'])
            self.append_stats('test', test_c)
            self.append_stats('test', test_e)
            if self.sample_fn is not None:
                self.samples['test'] = self.sample_fn(*inps)

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
        return train_c, train_e, train_o

    def add_monitor(self, k, extra=False):
        if extra:
            self.d[k] = {'mean': [], 'max': [], 'min': [], 'std': []}
        else:
            self.s[k] = []

    def append_s_stat(self, s_mean, s_max, s_min, s_std, stat):
        assert stat in self.d.keys()
        self.d[stat]['mean'].append(s_mean)
        self.d[stat]['max'].append(s_max)
        self.d[stat]['min'].append(s_min)
        self.d[stat]['std'].append(s_std)

    def save(self):
        plt.clf()
        x = 2
        y = ((len(self.params) + len(self.s) - 1) // 2) + 1
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
                    ax.fill_between(
                        range(len(stat_mean)),
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

    def disp(self, epoch, num, update_time):
        s = 'Epoch %d | sample %d | Update time: %.5f | ' % (epoch, num, update_time)
        for dataset, stats in self.stats.iteritems():
            if dataset == 'train':
                tag = ''
            else:
                tag = dataset + '_'
            for k, v in stats.iteritems():
                s += '%s%s: %.5f | ' % (tag, k, v[-1])
        print s
        for dataset, samples in self.samples.iteritems():
            print '%s-----------------' % dataset
            for k, v in samples.iteritems():
                if k in ['gt', 'es', 'xs']:
                    v = self.data['train'].dataset.translate(v)
                print '  %s: %s' % (k, v)

    def report(self):
        """Reports according to the best validation error score."""
        for key in self.err_keys:
            s = 'Best {} wrt validation: '.format(key)
            min_idx = np.argmin(self.errs['valid'][key])
            s += 'train: %.5f ' % self.errs['train'][key][min_idx]
            s += 'valid: %.5f ' % self.errs['valid'][key][min_idx]
            s += 'test: %.5f ' % self.errs['test'][key][min_idx]
            self.hyperparams['best_train_%s_wrt_valid' % key] =\
                self.errs['train'][key][min_idx]
            self.hyperparams['best_valid_%s_wrt_valid' % key] =\
                self.errs['valid'][key][min_idx]
            self.hyperparams['best_test_%s_wrt_valid' % key] =\
                self.errs['test'][key][min_idx]
            print s

    def save_best_model(self):
        """Save best models to disk."""
        s = signal.signal(signal.SIGINT, signal.SIG_IGN)
        hparams = self.hyperparams
        outdir = hparams['saveto']
        timestamp = self.timestamp

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        # save model parameters
        if self.early_stopping:
            for i in xrange(len(self.err_keys)):
                outfile = os.path.join(
                    outdir, 'model_{}_{}.npz'.format(
                        self.err_keys[i], timestamp))
                np.savez(outfile, **self.best_models[i])
        else:
            outfile = os.path.join(outdir,
                                   '{name}_{t}.npz'.format(name=self.name,
                                                           t=timestamp))
            np.savez(outfile, **dict((k, v.get_value())
                                  for k, v in self.tparams.items()))

        # save timings and model config
        #pkl.dump([self.costs, self.errs], open(
        #    os.path.join(outdir, 'timing_{}.pkl'.format(timestamp)), 'w'))
        #pkl.dump(hparams, open(
        #    os.path.join(outdir, 'config_{}.pkl'.format(timestamp)), 'w'))
        #signal.signal(signal.SIGINT, s)

    def _track_current_model(self, errs):
        """Keep track of best model and record it if necessary."""
        best_idx = self._update_best(errs)
        for i in xrange(len(best_idx)):
            if best_idx[i]:
                for k, v in self.tparams.items():
                    self.best_models[i][k] = self.tparams[k].get_value()

    def _validate(self, data_iter):
        """Iterate over validation/test set and get average costs."""
        costs_list = []
        errs_list = []
        while True:
            try:
                inps = data_iter.next()
            except StopIteration:
                break
            costs, outs, errs = self.split_outs(self.f_outs(*inps))
            costs_list.append(costs)
            errs_list.append(errs)
        costs = np.mean(np.asarray(costs_list), axis=0)
        errs = np.mean(np.asarray(costs_list), axis=0)
        return costs, outs, errs

    def _update_best(self, errs):
        """Update internal best list and return changed idx."""
        best_idx = np.zeros_like(errs, dtype=bool)
        for i in xrange(len(self.err_keys)):
            if errs[i] < self.best_errs[i]:
                self.best_errs[i] = errs[i]
                best_idx[i] = True
        return best_idx
