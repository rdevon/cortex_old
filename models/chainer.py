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
        outs = chain_dict['h_chains'] + [chain_dict['i_chain'],
                                         chain_dict['p_chain'],
                                         chain_dict['x_chain']]
        self.f_test = theano.function([X], chain_dict['extra'])
        self.f_build = theano.function([X], outs, updates=updates)

    def __call__(self, X, **chain_args):
        return self.assign(X, **chain_args)

    def build_data_chain(self, data_iter, batch_size=1, l_chain=None, h0s=None, c=None):
        n_remaining_samples = data_iter.n - data_iter.pos
        if l_chain is None:
            l_chain = n_remaining_samples
        else:
            l_chain = min(l_chain, n_remaining_samples)

        x = data_iter.next(batch_size=l_chain)[data_iter.name]
        x = np.zeros((x.shape[0], batch_size, x.shape[1])).astype(floatX) + x[:, None, :]
        for b in range(batch_size):
            np.random.shuffle(x[:, b])
        outs = self.f_build(x)
        h_chains = outs[:-3]
        idx, p_chain, x_chain = outs[-3:]

        rval = OrderedDict(
            idx=idx,
            x_chain=x_chain,
            p_chain=p_chain,
            h_chains=h_chains
        )

        return rval

    def step_energy(self, x, x_p, h_ps, *params):
        outs = self.rnn.step_sample(*(h_ps + [x_p] + list(params)))
        hs = outs[:self.rnn.n_layers]
        x_s, p = outs[self.rnn.n_layers:]
        energy    = self.rnn.neg_log_prob(x, p)
        #energy    = ((x - x_p) ** 2).sum(axis=x.ndim-1)
        return energy, hs, p

    def step_scale(self, scaling, counts, idx, alpha, beta):
        counts         = T.set_subtensor(counts[idx, T.arange(counts.shape[1])], 1)
        picked_scaling = scaling[idx, T.arange(scaling.shape[1])]
        scaling        = scaling / beta
        scaling        = T.set_subtensor(
            scaling[idx, T.arange(scaling.shape[1])], picked_scaling * alpha)
        scaling        = T.clip(scaling, 0.0, 1.0)
        return scaling, counts

    def step_assign(self, *params):
        params = list(params)
        h_ps = params[:self.rnn.n_layers]
        idx, e_p, counts, scaling, x, alpha, beta = params[self.rnn.n_layers:self.rnn.n_layers+7]
        params = params[self.rnn.n_layers+7:]
        x_p = x[idx, T.arange(x.shape[1])]

        if self.X_mean is not None:
            x_p = x_p - self.X_mean

        Us, h_ns, p = self.step_energy(x, x_p, h_ps, *params)
        Us         = Us - T.log(scaling)

        idx        = T.argmin(Us, axis=0)
        e_n        = Us[idx, T.arange(Us.shape[1])]

        scaling, counts = self.step_scale(scaling, counts, idx, alpha, beta)
        return tuple(h_ns) + (idx, e_n, p, counts, scaling), theano.scan_module.until(T.all(counts))

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

    def step_assign_call(self, X, h0s, condition_on, alpha, beta, steps, sample,
                         select_first, n_steps, *params):

        o_params  = self.rnn.get_output_args(*params)
        idx0 = T.zeros((X.shape[1],)).astype(intX)
        x0 = X[idx0, T.arange(X.shape[1])]
        #h0 = self.aux_net.feed(x0)

        if h0s is None:
            h0s = [T.zeros((X.shape[1], dim_h)).astype(floatX) for dim_h in self.rnn.dim_hs]
        p0 = self.rnn.output_net.feed(h0s[-1])

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
            outputs_info = h0s + [idx0, e0, None, counts, scaling]
            non_seqs = [X, alpha, beta]

            if condition_on is None:
                step = self.step_assign
            else:
                step = self.step_assign_cond
                non_seqs.append(condition_on)

            non_seqs += params

            outs, updates = scan(
                step, seqs, outputs_info, non_seqs, steps, name='make_chain',
                strict=False)

            h_chains = outs[:self.rnn.n_layers]
            (i_chain, energies, p_chain, counts, scalings) = outs[self.rnn.n_layers:]

            h_chains = [concatenate([h0[None, :, :], h_chain], axis=0)
                        for h0, h_chain in zip(h0s, h_chains)]
            i_chain = concatenate([idx0[None, :], i_chain], axis=0).astype('int64')
            p_chain = concatenate([p0[None, :, :], p_chain], axis=0)
            x_chain = X[i_chain, T.arange(X.shape[1])]

            outs = OrderedDict(
                extra=p0,
                i_chain=i_chain,
                p_chain=p_chain,
                h_chains=h_chains,
                x_chain=x_chain,
                energies=energies,
                counts=counts[-1],
                scalings=scalings[-1]
            )

        return outs, updates

    def assign(self, X, h0s=None, condition_on=None, alpha=0.0, beta=1.0,
               steps=None, sample=False, select_first=False, n_steps=None):

        if steps is None:
            steps = X.shape[0] - 1

        params = self.rnn.get_sample_params()

        return self.step_assign_call(X, h0s, condition_on, alpha, beta, steps,
                                     sample, select_first, n_steps, *params)


class DijkstrasChainer(RNNChainer):
    def __init__(self, rnn):
        self.rnn = rnn

        X   = T.matrix('x', dtype=floatX)
        Hs  = [T.matrix('h%d' % i, dtype=floatX) for i in range(self.rnn.n_layers)]
        idx = T.vector('idx', dtype=intX)

        chain_dict, updates = self.__call__(X, Hs, idx)
        outs = chain_dict['h_chains'] + [chain_dict['x_chain'],
                                         chain_dict['i_chain'],
                                         chain_dict['distances'],
                                         chain_dict['mask']]
        self.f_test = theano.function(Hs + [X, idx], chain_dict['extra'],
                                      on_unused_input='ignore')
        self.f_build = theano.function(Hs + [X, idx], outs, updates=updates,
                                       profile=False)

    def __call__(self, X, Hs, idx):
        return self.assign(X, Hs, idx)

    def build_data_chain(self, data_iter, build_batch=100):
        '''Build a chain or real data.

        Pulls a batch out of a Dataset class instance and performs all-pairs
        shortest path (APSP) using Dijkstra's algorithm.

        Args:
            data_iter: Dataset class instance.
            batch_size: int.
            build_batch: int (optional). Number of SSSP problems to solve at
                once.

        Returns:
            rval: OrderedDict. mask, x_chain, i_chain, and h_chains.

        '''
        outs = data_iter.next()
        x = outs[data_iter.name]
        build_batch = min(x.shape[0], build_batch)
        hs = outs['hs']
        pos = outs['pos']
        data_idx = outs['idx']

        x_chain = []
        i_chain = []
        hs_chain = [[] for _ in hs]
        masks = []
        ds = []
        for i0 in range(0, x.shape[0] - build_batch + 1, build_batch):
            #print 'here', i0, build_batch, hs[0].shape, x.shape, x[range(i0, i0 + build_batch)].shape
            inps = hs + [x, range(i0, i0 + build_batch)]
            #assert False, self.f_test(*inps)
            outs = self.f_build(*inps)
            #print 'build done'
            hs_ = outs[:-4]
            x_, idx, d, mask = outs[-4:]
            ds.append(d)
            x_chain.append(x_)
            i_chain.append(idx)
            #print x[range(i0, i0 + build_batch)]
            #print x_.shape
            #print x_[:, 0, 1]
            #assert False, idx
            for i, h_ in enumerate(hs_):
                hs_chain[i].append(h_)
            masks.append(mask)

        max_length = max(m.shape[0] for m in masks)
        for i, mask in enumerate(masks):

            npad = ((0, max_length - mask.shape[0]), (0, 0), (0, 0))

            masks[i] = np.pad(masks[i],
                              pad_width=npad[:-1],
                              mode='constant',
                              constant_values=0)
            x_chain[i] = np.pad(x_chain[i], pad_width=npad, mode='constant', constant_values=0)
            i_chain[i] = np.pad(i_chain[i], pad_width=npad[:-1], mode='constant', constant_values=0)
            for j, h_chain in enumerate(hs_chain):
                hs_chain[j][i] = np.pad(h_chain[i], pad_width=npad, mode='constant', constant_values=0)

        mask     = np.concatenate(masks, axis=1).astype(floatX)
        x_chain  = np.concatenate(x_chain, axis=1).astype(floatX)
        i_chain  = np.concatenate(i_chain, axis=1).astype(intX) + pos
        h_chains = [np.concatenate(h_chain, axis=1) for h_chain in hs_chain]
        distances = np.concatenate(ds, axis=0).astype(floatX)

        '''
        mask = mask.reshape((mask.shape[0], mask.shape[1] * mask.shape[2]))

        x_chain  = x_chain.reshape(
            (x_chain.shape[0],
             x_chain.shape[1] * x_chain.shape[2],
             x_chain.shape[3]))

        i_chain  = i_chain.reshape(
            i_chain.shape[0], i_chain.shape[1] * i_chain.shape[2])

        h_chains = [h_chain.reshape(
            (h_chain.shape[0],
             h_chain.shape[1] * h_chain.shape[2],
             h_chain.shape[3]))
                    for h_chain in h_chains]
        '''

        #print distances.shape, x_chain.shape, mask.shape
        #distances = distances.reshape((distances.shape[0] * distances.shape[1],))
        #print distances.min(), distances.max(), distances.std(), distances.mean()
        #assert False#, np.where(distances == 0)[0].tolist()

        rval = OrderedDict(
            mask=mask,
            x_chain=x_chain,
            i_chain=i_chain,
            h_chains=h_chains,
            data_idx=data_idx
        )
        return rval

    def step_assign_call(self, X, Hs, idx, *params):

        def step(i, D, c, E):
            c = T.switch(T.eq(i, 0), 0, c)
            D_new = T.switch(T.lt(E[i][:, None] + D[i][None, :], D), E[i][:, None] + D[i][None, :], D)
            c = T.switch(T.all(T.eq(D, D_new)), c, c + 1)
            stop = T.eq(c, 0) and T.eq(i, D.shape[0] - 1)
            return (D_new, c), theano.scan_module.until(stop)

        '''
        def step(i, D, E):
            D_new = T.switch(T.lt(E[i][:, None] + D[i][None, :], D), E[i][:, None] + D[i][None, :], D)
            return D_new, theano.scan_module.until(T.all(T.eq(D, D_new)))
        '''

        def step_path(i, Q, Ds):
            P = T.switch(T.eq(Ds[i], Ds[i-1]), -1, i % Ds.shape[1])
            Q = T.switch(T.neq(P, -1), P, Q)
            return Q, P

        # Step forward in RNN and calculate energies.
        if True:
            outs = self.rnn.step_sample(*(Hs + [X] + list(params)))
            P = outs[-1]
            E = self.rnn.neg_log_prob(X[:, None, :], P[None, :, :])
        else:
            E = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        D = E[idx].T
        #D *= self.rnn.trng.normal(size=D.shape, dtype=D.dtype)

        C = T.tile(T.arange(D.shape[0]), D.shape[0])

        (Ds, Cs), updates = theano.scan(
            step,
            sequences=[C],
            outputs_info=[D, T.constant(0)],
            non_sequences=[E]
        )

        Q0 = T.zeros_like(D) + T.arange(D.shape[0])[:, None]
        (Qs, Ps), updates = theano.scan(
            step_path,
            sequences=[T.arange(1, Ds.shape[0])[::-1]],
            outputs_info=[Q0, None],
            non_sequences=[Ds],
            name='path'
        )

        # Append start and ends
        Qs0 = T.zeros_like(D) + T.arange(D.shape[1])[None, :]
        Qsl = T.zeros_like(D) + T.arange(D.shape[0])[:, None]
        Qs = T.concatenate([Qs0[None, :, :], Qs[::-1], Qsl[None, :, :]]).astype(intX)
        Ps = T.concatenate([Qs0[None, :, :], Ps[::-1], Qsl[None, :, :]])

        # Keep chains with short distances and flatten across start / end axes
        Ds = Ds[-1]
        Ds = Ds.reshape((Ds.shape[0] * Ds.shape[1],))
        #c_idx = T.and_(T.lt(Ds, 2 * Ds.max()), T.neq(Ds, 0)).nonzero()[0]
        #Ds = Ds[c_idx]
        Qs = Qs.reshape((Qs.shape[0], Qs.shape[1] * Qs.shape[2]))#[:, c_idx]
        Ps = Ps.reshape((Ps.shape[0], Ps.shape[1] * Ps.shape[2]))#[:, c_idx]

        # Compress and remove chain steps with no changes

        def step_unique(Q, P):
            idx = T.neq(P, -1).nonzero()[0]
            Q_ = T.zeros_like(Q)
            Q = T.set_subtensor(Q_[:idx.shape[0]], Q[idx])
            P_ = T.zeros_like(P) - 1
            P = T.set_subtensor(P_[:idx.shape[0]], P[idx])
            return Q, P

        (Qs, Ps), _ = theano.scan(
            step_unique,
            sequences=[Qs.T, Ps.T],
            outputs_info=[None, None],
            name='unique'
        )
        Qs = Qs.T
        Ps = Ps.T

        idx = T.neq(Ps.sum(axis=1), -(Ps.shape[1])).nonzero()[0]
        Qs = Qs[idx]
        Ps = Ps[idx]

        # Build X chain
        x_chain = X[Qs]
        x_chain = T.switch(T.eq(Ps[:, :, None], -1), 0, x_chain)
        i_chain = Qs
        h_chains = [H[Qs] for H in Hs]
        mask = T.neq(Ps, -1).astype(intX)

        outs = OrderedDict(
            extra=(Ps, Qs, mask, x_chain),
            mask=mask,
            x_chain=x_chain,
            i_chain=i_chain,
            distances=Ds,
            h_chains=h_chains,
        )

        return outs, theano.OrderedUpdates()

    def assign(self, X, Hs, idx):
        params = self.rnn.get_sample_params()
        return self.step_assign_call(X, Hs, idx, *params)


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