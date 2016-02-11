'''
Module for the Chainer model
'''

from collections import OrderedDict
import numpy as np
from progressbar import ProgressBar, Timer
import theano
from theano import tensor as T
import time

from . import Layer
from utils.tools import (
    concatenate,
    floatX,
    scan
)


class RNNChainer(Layer):
    def __init__(self, rnn, **chain_args):
        self.rnn = rnn
        X = T.tensor3('x', dtype=floatX)
        H0 = T.matrix('h0', dtype=floatX)
        chain_dict, updates = self.__call__(X, H0=H0, **chain_args)
        outs = [chain_dict['i_chain'], chain_dict['p_chain'], chain_dict['h_chain']]
        self.f_build = theano.function([X, H0], outs, updates=updates)

    def __call__(self, X, H0=None, C=None, **chain_args):
        chain_dict, updates = self.assign(X, H0, condition_on=C, **chain_args)
        return chain_dict, updates

    def build_data_chain(self, data_iter, l_chain=None, h0=None, c=None):
        n_remaining_samples = data_iter.n - data_iter.pos
        if l_chain is None:
            l_chain = n_remaining_samples
        else:
            l_chain = min(l_chain, n_remaining_samples)

        x = data_iter.next(batch_size=l_chain)[data_iter.name][:, None, :]
        if h0 is None:
            h0 = np.tanh(self.rnn.rng.uniform(
                low=-2., high=2., size=(1, self.rnn.dim_h,)).astype(floatX))

        widgets = ['Building chain from dataset {dataset} '
                   'of length {length} ('.format(dataset=data_iter.name,
                                               length=l_chain),
                   Timer(), ')']
        pbar = ProgressBar(widgets=widgets, maxval=1).start()
        idx, p_chain, h_chain = self.f_build(x, h0)
        pbar.update(1)
        x_chain = x[idx]

        rval = OrderedDict(
            idx=idx,
            x_chain=x_chain,
            p_chain=p_chain,
            h_chain=h_chain
        )

        return rval

        # Assignment functions -----------------------------------------------------

    def step_energy(self, x, x_p, h_p, *params):
        h, x_s, p = self.rnn.step_sample(h_p, x_p, *params)
        energy = self.rnn.neg_log_prob(x, p)
        return energy, h, p

    def step_energy_cond(rnn, x, x_p, h_p, c, *params):
        h, x_s, p = self.step_sample_cond(h_p, x_p, c, *params)
        energy = self.rnn.neg_log_prob(x, p)
        return energy, h, p

    def step_scale(self, scaling, counts, idx, alpha, beta):
        counts = T.set_subtensor(counts[idx, T.arange(counts.shape[1])], 1)
        picked_scaling = scaling[idx, T.arange(scaling.shape[1])]
        scaling = scaling / beta
        scaling = T.set_subtensor(scaling[idx, T.arange(scaling.shape[1])], picked_scaling * alpha)
        scaling = T.clip(scaling, 0.0, 1.0)
        return scaling, counts

    def step_assign(self, idx, h_p, counts, scaling, x, alpha, beta, *params):
        energies, h_n, p = self.step_energy(x, x[idx, T.arange(x.shape[1])], h_p, *params)
        energies -= T.log(scaling)

        idx = T.argmin(energies, axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, p, energies[idx, T.arange(energies.shape[1])], counts, scaling), theano.scan_module.until(T.all(counts))

    def step_assign_cond(self, idx, h_p, counts, scaling, x, alpha, beta, c, *params):
        energies, h_n, p = self.step_energy_cond(x, x[idx, T.arange(x.shape[1])], h_p, c, *params)
        energies -= T.log(scaling)

        idx = T.argmin(energies, axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, p, energies[idx, T.arange(energies.shape[1])], counts, scaling), theano.scan_module.until(T.all(counts))

    def step_assign_sample(self, idx, h_p, counts, scaling, x, alpha, beta, *params):
        energies, h_n, p = self.step_energy(x, x[idx, T.arange(x.shape[1])], h_p, *params)
        energies -= T.log(scaling)

        e_max = (-energies).max()
        probs = T.exp(-energies - e_max)
        probs = probs
        probs = probs / probs.sum()
        idx = T.argmax(self.rnn.trng.multinomial(pvals=probs).astype('int64'), axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, p, energies[idx, T.arange(energies.shape[1])], counts, scaling), theano.scan_module.until(T.all(counts))

    def get_first_assign(self, x, p0):
        energy = self.rnn.neg_log_prob(x, p0)
        idx = T.argmin(energy, axis=0)
        return idx

    def step_assign_call(self, X, h0, condition_on, alpha, beta, steps, sample,
                         select_first, *params):

        o_params = self.rnn.get_output_args(*params)
        p0 = self.rnn.output_net.feed(h0)

        counts = T.zeros((X.shape[0], X.shape[1])).astype('int64')
        scaling = T.ones((X.shape[0], X.shape[1])).astype('float32')

        if select_first:
            print 'Selecting first best in assignment'
            idx0 = self.get_first_assign(X, p0)
        else:
            print 'Using 0 as first in assignment'
            idx0 = T.zeros((X.shape[1],)).astype('int64')

        counts = T.set_subtensor(counts[idx0, T.arange(counts.shape[1])], 1)
        scaling = T.set_subtensor(scaling[idx0, T.arange(scaling.shape[1])], scaling[0] * alpha)

        seqs = []
        outputs_info = [idx0, h0, None, None, counts, scaling]
        non_seqs = [X, alpha, beta]

        if condition_on is None:
            step = self.step_assign
        else:
            step = self.step_assign_cond
            non_seqs.append(condition_on)

        non_seqs += params

        (i_chain, h_chain, p_chain, energies, counts, scalings), updates = scan(
            step, seqs, outputs_info, non_seqs, steps, name='make_chain',
            strict=False)

        i_chain = concatenate([idx0[None, :], i_chain], axis=0)
        p_chain = concatenate([p0[None, :, :], p_chain], axis=0)

        outs = OrderedDict(
            i_chain=i_chain.astype('int64'),
            p_chain=p_chain,
            h_chain=h_chain,
            energies=energies,
            counts=counts[-1],
            scalings=scalings[-1]
        )

        return outs, updates

    def assign(self, X, h0=None, condition_on=None, alpha=0.0, beta=1.0,
               steps=None, sample=False, select_first=False):
        if h0 is None:
            h0 = T.tanh(self.rnn.trng.uniform(
                low=-2., high=2., size=(X.shape[1], self.rnn.dim_h))).astype(floatX)

        if steps is None:
            steps = X.shape[0] - 1

        params = self.rnn.get_sample_params()

        return self.step_assign_call(X, h0, condition_on, alpha, beta, steps,
                                     sample, select_first, *params)