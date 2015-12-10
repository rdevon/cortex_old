'''
Chains dataset.
'''

import numpy as np
from os import path
from progressbar import ProgressBar
import random
import theano
from theano import tensor as T
import time

from utils.tools import (
    concatenate,
    init_rngs,
    rng_,
    scan
)


def energy(x, x_p, h_p, model):
    params = model.get_sample_params()
    h, x_s, p = model.step_sample(h_p[None, :], x_p[None, :], *params)
    energy = model.neg_log_prob(x, p)
    return energy, x_s[0], h[0]

def distance(x, x_p, h_p, model):
    distance = (x - x_p[None, :]) ** 2
    distance = distance.sum(axis=1)
    return distance, x, h_p

def random_distance(x, x_p, h_p, model):
    distance = model.trng.uniform(size=(x.shape[0],), dtype=x_p.dtype)
    return distance, x, h_p


class Chains(object):
    def __init__(self, D, batch_size=10,
                 window=20, chain_length=5000, build_batch=1000,
                 chain_stride=None, chain_noise=0.,
                 use_theano=False,
                 trim_end=0, out_path=None, **kwargs):

        self.dataset = D(**kwargs)

        self.batch_size = batch_size
        self.f_energy = None
        self.f_chain = None
        self.window = window
        self.build_batch = build_batch
        self.trim_end = trim_end
        self.chain_length = min(chain_length, self.dataset.n)
        self.chain_noise = chain_noise
        self.dim_h = None
        self.out_path = out_path
        self.use_theano = use_theano

        if chain_stride is None:
            self.chain_stride = self.window
        else:
            self.chain_stride = chain_stride

        self.next = self._next
        self.cpos = -1
        init_rngs(self, **kwargs)

    def next(self):
        raise NotImplementedError()

    def set_f_energy(self, f_energy, dim_h, model=None,
                     alpha=0.0, beta=1.0, steps=None):
        self.dim_h = dim_h

        # Energy function -----------------------------------------------------
        X = T.matrix('x', dtype='float32')
        x_p = T.vector('x_p', dtype='float32')
        h_p = T.vector('h_p', dtype='float32')

        energy, x_n, h_n = f_energy(X, x_p, h_p, model)
        self.f_energy = theano.function([X, x_p, h_p], [energy, x_n, h_n])

        # Chain function ------------------------------------------------------
        counts = T.zeros((X.shape[0],)).astype('int64')
        scaling = T.ones((X.shape[0],)).astype('float32')
        P = T.scalar('P', dtype='int64')
        S = T.scalar('S', dtype='float32')
        x_p = X[0]
        h_p = self.trng.normal(avg=0., std=1., size=(dim_h,)).astype('float32')
        counts = T.set_subtensor(counts[0], 1)
        scaling = T.set_subtensor(scaling[0], scaling[0] * alpha)

        if steps is None:
            steps = X.shape[0] - 1

        chain_noise = self.trng.normal(avg=0., std=S, size=(steps,)).astype('floatX')

        def f_step(cn, i, x_p, h_p, counts, scaling, x):
            energies, _, h_n = f_energy(x, x_p, h_p, model)
            energies = (energies / scaling) + cn
            i = T.argmin(energies)
            counts = T.set_subtensor(counts[i], 1)
            picked_scaling = scaling[i]
            scaling = scaling / beta
            scaling = T.set_subtensor(scaling[i], picked_scaling * alpha)
            scaling = T.clip(scaling, 0.0, 1.0)
            return (i, x[i], h_n, counts, scaling), theano.scan_module.until(T.all(counts))

        seqs = [chain_noise]
        outputs_info = [T.constant(0).astype('int64'), x_p, h_p, counts, scaling]
        non_seqs = [X]

        (chain, x_chain, h_chain, counts, scalings), updates = scan(
            f_step, seqs, outputs_info, non_seqs, steps, name='make_chain',
            strict=False)
        counts = counts[-1]

        chain += P
        chain_e = T.zeros((chain.shape[0] + 1,)).astype('int64')
        chain = T.set_subtensor(chain_e[1:], chain)
        perc = counts.sum() / counts.shape[0].astype('float32')
        self.f_chain = theano.function([X, P, S], [chain, perc], updates=updates)

    def _build_chain_py(self, x, data_pos):
        h_p = np.random.normal(
            loc=0, scale=1, size=(self.dim_h,)).astype('float32')

        l_chain = x.shape[0]
        n_samples = min(self.build_batch, l_chain)
        chain_idx = range(l_chain)
        rnd_idx = np.random.permutation(np.arange(0, l_chain, 1))
        chain_idx = [chain_idx[i] for i in rnd_idx]

        counts = [True for _ in xrange(l_chain)]
        n = l_chain
        chain = []

        pbar = ProgressBar(maxval=l_chain).start()
        while n > 0:
            idx = [j for j in chain_idx if counts[j]]
            x_idx = [idx[i] for i in range(min(n_samples, n))]
            assert len(np.unique(x_idx)) == len(x_idx)

            if n == l_chain:
                picked_idx = random.choice(x_idx)
            else:
                assert x_p is not None
                x_n = x[x_idx]
                energies, _, h_p = self.f_energy(x_n, x_p, h_p)
                i = np.argmin(energies)
                picked_idx = x_idx[i]

            counts[picked_idx] = False
            assert not (picked_idx + data_pos) in self.chain
            chain.append(picked_idx + data_pos)

            x_p = x[picked_idx]
            n -= 1

            nd_idx = np.random.permutation(np.arange(0, l_chain, 1))
            chain_idx = [chain_idx[i] for i in rnd_idx]

            pbar.update(l_chain - n)

        return chain

    def _build_chain(self, trim_end=0, use_noise=True):
        self.chain = []
        n_remaining_samples = self.dataset.n - self.dataset.pos
        l_chain = min(self.chain_length, n_remaining_samples)

        data_pos = self.dataset.pos
        x, _ = self.dataset.next(batch_size=l_chain)
        n_samples = min(self.build_batch, l_chain)

        t0 = time.time()
        if self.use_theano:
            print('Resetting chain. Position in data is %d' % (data_pos))
            if use_noise:
                s = self.chain_noise
            else:
                s = 0.
            self.chain, perc = self.f_chain(x, data_pos, s)
            print 'Chain has length %d and used %.2f percent of data points' % (len(self.chain), 100 * perc)
        else:
            print('Resetting chain with length %d and %d samples per query. '
                  'Position in data is %d'
                  % (l_chain, n_samples, data_pos))
            self.chain = self._build_chain_py(x, data_pos)
        t1 = time.time()
        print 'Chain took %.2f seconds' % (t1 - t0)

        if self.out_path is not None:
            self.dataset.save_images(
                self._load_chains(),
                path.join(self.out_path,
                          '%s_chain_%d.png'
                          % (self.dataset.mode, self.dataset.pos)),
                x_limit=200)

        if trim_end:
            print 'Trimming %d' % trim_end
            self.chain = self.chain[:-trim_end]

            if self.out_path is not None:
                self.dataset.save_images(
                    self._load_chains(),
                    path.join(self.out_path,
                              '%s_chain_%d_trimmed.png'
                              % (self.dataset.mode, self.dataset.pos)),
                    x_limit=200)

    def _load_chains(self, chains=None):
        if chains is None:
            chains = [self.chain]

        x = np.zeros((len(chains[0]),
                      len(chains),
                      self.dataset.dim)
            ).astype('float32')
        for i, c in enumerate(chains):
            x[:, i] = self.dataset.X[c]
        return x

    def _get_labels(self, chains=None):
        if chains is None:
            chains = [self.chain]

        y = []
        for chain in chains:
            y_ = []
            for c in chain:
                y_.append(self.dataset.Y[c])
            y.append(y_)
        return np.array(y).astype('float32')

    def reset(self):
        self.dataset.reset()
        self.cpos = -1

    def _next(self, l_chain=None, use_noise=True):
        assert self.f_energy is not None

        chain_length = min(self.chain_length - self.trim_end,
                           self.dataset.n - self.dataset.pos - self.trim_end)
        window = min(self.window, chain_length)

        if self.cpos == -1:
            self.cpos = 0
            self._build_chain(trim_end=self.trim_end, use_noise=True)
            self.chain_idx = range(0, chain_length - window + 1, self.chain_stride)
            random.shuffle(self.chain_idx)

        chains = []
        for b in xrange(self.batch_size):
            p = self.chain_idx[b + self.cpos]
            chains.append([self.chain[j] for j in xrange(p, p + window)])

        x = self._load_chains(chains=chains)

        if self.cpos + 2 * self.batch_size >= len(self.chain_idx):
            self.cpos = -1
        else:
            self.cpos += self.batch_size

        return x

    def next_simple(self, batch_size=None):
        x = self.dataset.next(batch_size=batch_size)
        return x