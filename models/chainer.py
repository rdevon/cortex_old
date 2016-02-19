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
from utils import floatX, intX
from utils.tools import (
    concatenate,
    scan
)


class RNNChainer(Layer):
    def __init__(self, rnn, X_mean=None, aux_net=None, **chain_args):
        self.rnn = rnn
        self.aux_net = aux_net
        self.X_mean = X_mean

        X = T.tensor3('x', dtype=floatX)

        chain_dict, updates = self.__call__(X, **chain_args)
        outs = [chain_dict['i_chain'], chain_dict['p_chain'],
                chain_dict['h_chain'], chain_dict['x_chain']]
        self.f_test = theano.function([X], chain_dict['extra'])
        self.f_build = theano.function([X], outs, updates=updates)

    def __call__(self, X, **chain_args):
        return self.assign(X, **chain_args)

    def build_data_chain(self, data_iter, l_chain=None, h0=None, c=None):
        n_remaining_samples = data_iter.n - data_iter.pos
        if l_chain is None:
            l_chain = n_remaining_samples
        else:
            l_chain = min(l_chain, n_remaining_samples)

        x = data_iter.next(batch_size=l_chain)[data_iter.name][:, None, :]

        widgets = ['Building chain from dataset {dataset} '
                   'of length {length} ('.format(dataset=data_iter.name,
                                               length=l_chain),
                   Timer(), ')']
        pbar = ProgressBar(widgets=widgets, maxval=1).start()
        idx, p_chain, h_chain, x_chain = self.f_build(x)
        pbar.update(1)
        print
        idx = idx[:, 0]
        p_chain = p_chain[:, 0]
        h_chain = h_chain[:, 0]
        x_chain = x_chain[:, 0]

        rval = OrderedDict(
            idx=idx,
            x_chain=x_chain,
            p_chain=p_chain,
            h_chain=h_chain
        )

        return rval

    def step_energy(self, x, x_p, h_p, *params):
        h, x_s, p = self.rnn.step_sample(h_p, x_p, *params)
        energy    = self.rnn.neg_log_prob(x, p)
        #energy    = ((x - x_p) ** 2).sum(axis=x.ndim-1)
        return energy, h, p

    '''
    def step_energy_cond(rnn, x, x_p, h_p, c, *params):
        h, x_s, p = self.step_sample_cond(h_p, x_p, c, *params)
        energy = self.rnn.neg_log_prob(x, p)
        return energy, h, p
    '''

    def step_scale(self, scaling, counts, idx, alpha, beta):
        counts         = T.set_subtensor(counts[idx, T.arange(counts.shape[1])], 1)
        picked_scaling = scaling[idx, T.arange(scaling.shape[1])]
        scaling        = scaling / beta
        scaling        = T.set_subtensor(
            scaling[idx, T.arange(scaling.shape[1])], picked_scaling * alpha)
        scaling        = T.clip(scaling, 0.0, 1.0)
        return scaling, counts

    def step_assign(self, idx, e_p, h_p, counts, scaling, x, alpha, beta, *params):
        x_p = x[idx, T.arange(x.shape[1])]

        if self.X_mean is not None:
            x_p = x_p - self.X_mean

        Us, h_n, p = self.step_energy(x, x_p, h_p, *params)
        Us         = Us - T.log(scaling)

        idx        = T.argmin(Us, axis=0)
        #V = x - x_p
        #KEs = (V ** 2 * self.rnn.M[None, None, :]).sum(axis=2)
        #idx = T.argmin(abs(Us + KEs - e_p[None, :]), axis=0)
        #e_n         = (Us + KEs)[idx, T.arange(Us.shape[1])]
        e_n        = Us[idx, T.arange(Us.shape[1])]

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, e_n, h_n, p, counts, scaling), theano.scan_module.until(T.all(counts))

    def step_assign_sample(self, r, h_p, x_p, x, *params):
        # assigning
        Us, h_a, p_a  = self.step_energy(x, x_p, h_p, *params)
        idx           = T.argmin(Us, axis=0)
        x_a           = x[idx, T.arange(x.shape[1])]

        # sampling
        h_s, x_s, p_s = self.rnn.step_sample(h_p, x_p, *params)
        x_n = T.switch(T.lt(r[:, None], 0.1), x_a, x_s)
        h_n = T.switch(T.lt(r[:, None], 0.1), h_a, h_s)
        p_n = T.switch(T.lt(r[:, None], 0.1), p_a, p_s)

        return h_n, x_n, p_n

    '''
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
    '''

    def get_first_assign(self, x, p0):
        energy = self.rnn.neg_log_prob(x, p0)
        idx = T.argmin(energy, axis=0)
        return idx

    def step_assign_call(self, X, h0, condition_on, alpha, beta, steps, sample,
                         select_first, n_steps, *params):

        o_params  = self.rnn.get_output_args(*params)
        idx0 = T.zeros((X.shape[1],)).astype(intX)
        x0 = X[idx0, T.arange(X.shape[1])]
        #h0 = self.aux_net.feed(x0)

        if h0 is None:
            h0 = T.zeros((X.shape[1], self.rnn.dim_h)).astype(floatX)
        p0 = self.rnn.output_net.feed(h0)

        if sample:
            assert n_steps is not None
            rs = self.rnn.trng.uniform((n_steps, X.shape[1])).astype(floatX)
            seqs = [rs]
            outputs_info = [h0, x0, None]
            non_seqs = [X] + list(params)
            step = self.step_assign_sample

            #return step(rs[0], h0, x0, X, *params)

            (h_chain, x_chain, p_chain), updates = scan(
                step, seqs, outputs_info, non_seqs, n_steps,
                name='make_chain_and_sample')

            h_chain = concatenate([h0[None, :, :], h_chain], axis=0)
            x_chain = concatenate([x0[None, :, :], x_chain], axis=0)
            p_chain = concatenate([p0[None, :, :], p_chain], axis=0)

            outs = OrderedDict(
                p_chain=p_chain,
                h_chain=h_chain,
                x_chain=x_chain
            )
        else:
            e0 = self.rnn.neg_log_prob(x0, p0)
            counts    = T.zeros((X.shape[0], X.shape[1])).astype('int64')
            scaling   = T.ones((X.shape[0], X.shape[1])).astype(floatX)

            counts  = T.set_subtensor(counts[idx0, T.arange(counts.shape[1])], 1)
            scaling = T.set_subtensor(scaling[idx0, T.arange(scaling.shape[1])],
                                      scaling[0] * alpha)

            seqs = []
            outputs_info = [idx0, e0, h0, None, counts, scaling]
            non_seqs = [X, alpha, beta]

            if condition_on is None:
                step = self.step_assign
            else:
                step = self.step_assign_cond
                non_seqs.append(condition_on)

            non_seqs += params

            (i_chain, energies, h_chain, p_chain, counts, scalings), updates = scan(
                step, seqs, outputs_info, non_seqs, steps, name='make_chain',
                strict=False)

            h_chain = concatenate([h0[None, :, :], h_chain], axis=0)
            i_chain = concatenate([idx0[None, :], i_chain], axis=0).astype('int64')
            p_chain = concatenate([p0[None, :, :], p_chain], axis=0)
            x_chain = X[i_chain, T.arange(X.shape[1])]

            outs = OrderedDict(
                extra=p0,
                i_chain=i_chain,
                p_chain=p_chain,
                h_chain=h_chain,
                x_chain=x_chain,
                energies=energies,
                counts=counts[-1],
                scalings=scalings[-1]
            )

        return outs, updates

    def assign(self, X, h0=None, condition_on=None, alpha=0.0, beta=1.0,
               steps=None, sample=False, select_first=False, n_steps=None):

        if steps is None:
            steps = X.shape[0] - 1

        params = self.rnn.get_sample_params()

        return self.step_assign_call(X, h0, condition_on, alpha, beta, steps,
                                     sample, select_first, n_steps, *params)


class LSTMChainer(RNNChainer):
    def step_energy(self, x, x_p, h_p, c_p, *params):
        h, c, x_s, p = self.rnn.step_sample(h_p, c_p, x_p, *params)
        energy    = self.rnn.neg_log_prob(x, p)
        #energy    = ((x - x_p) ** 2).sum(axis=x.ndim-1)
        return energy, h, c, p

    def step_assign(self, idx, h_p, c_p, counts, scaling, x, alpha, beta, *params):
        x_p = x[idx, T.arange(x.shape[1])]

        if self.X_mean is not None:
            x_p = x_p - self.X_mean

        energies, h_n, c_n, p = self.step_energy(x, x_p, h_p, c_p, *params)
        energies         = energies - T.log(scaling)
        energy           = energies[idx, T.arange(energies.shape[1])]
        idx              = T.argmin(energies, axis=0)

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return (idx, h_n, c_n, p, energy, counts, scaling), theano.scan_module.until(T.all(counts))

    def get_first_assign(self, x, p0):
        energy = self.rnn.neg_log_prob(x, p0)
        idx = T.argmin(energy, axis=0)
        return idx

    def step_assign_call(self, X, h0, c0, condition_on, alpha, beta, steps, sample,
                         select_first, *params):

        o_params  = self.rnn.get_output_args(*params)
        counts    = T.zeros((X.shape[0], X.shape[1])).astype('int64')
        scaling   = T.ones((X.shape[0], X.shape[1])).astype(floatX)
        constants = []

        idx0 = T.zeros((X.shape[1],)).astype(intX)
        x0 = X[idx0, T.arange(X.shape[1])]
        if h0 is None:
            h0 = T.zeros((X.shape[1], self.rnn.dim_h)).astype(floatX)
        if c0 is None:
            c0 = T.zeros((X.shape[1], self.rnn.dim_h)).astype(floatX)
        p0 = self.rnn.output_net.feed(h0)
        e0 = self.rnn.neg_log_prob(x0, p0)
        constants = []

        counts  = T.set_subtensor(counts[idx0, T.arange(counts.shape[1])], 1)
        scaling = T.set_subtensor(scaling[idx0, T.arange(scaling.shape[1])],
                                  scaling[0] * alpha)

        seqs = []
        outputs_info = [idx0, h0, c0, None, None, counts, scaling]
        non_seqs = [X, alpha, beta]

        if condition_on is None:
            step = self.step_assign
        else:
            step = self.step_assign_cond
            non_seqs.append(condition_on)

        non_seqs += params

        (i_chain, h_chain, c_chain, p_chain, energies, counts, scalings), updates = scan(
            step, seqs, outputs_info, non_seqs, steps, name='make_chain',
            strict=False)

        h_chain = concatenate([h0[None, :, :], h_chain], axis=0)
        i_chain = concatenate([idx0[None, :], i_chain], axis=0).astype('int64')
        p_chain = concatenate([p0[None, :, :], p_chain], axis=0)
        x_chain = X[i_chain, T.arange(X.shape[1])]

        outs = OrderedDict(
            extra=p0,
            i_chain=i_chain,
            p_chain=p_chain,
            h_chain=h_chain,
            x_chain=x_chain,
            energies=energies,
            counts=counts[-1],
            scalings=scalings[-1]
        )

        return outs, updates, constants

    def assign(self, X, h0=None, c0=None, condition_on=None, alpha=0.0, beta=1.0,
               steps=None, sample=False, select_first=False):

        if steps is None:
            steps = X.shape[0] - 1

        params = self.rnn.get_sample_params()

        return self.step_assign_call(X, h0, c0, condition_on, alpha, beta, steps,
                                     sample, select_first, *params)
