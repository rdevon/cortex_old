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
    floatX,
    init_rngs,
    rng_,
    scan
)


def energy(x, x_p, h_p, model):
    params = model.get_sample_params()
    h, x_s, p = model.step_sample(h_p[None, :], x_p[None, :], *params)
    energy = model.neg_log_prob(x, p)
    return energy, x_s[0], h[0]

def distance(x, x_p, h_p):
    distance = (x - x_p[None, :]) ** 2
    distance = distance.sum(axis=1)
    return distance, x, h_p

def random_distance(x, x_p, h_p):
    raise NotImplementedError()
    distance = model.trng.uniform(size=(x.shape[0],), dtype=x_p.dtype)
    return distance, x, h_p


class Chains(object):
    def __init__(self, D, data_name, batch_size=10,
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
        self.dim = self.dataset.dims[data_name]
        self.mean_image = self.dataset.mean_image
        self.save_images = self.dataset.save_images

        if chain_stride is None:
            self.chain_stride = self.window
        else:
            self.chain_stride = chain_stride

        self.next = self._next
        self.cpos = -1
        init_rngs(self, **kwargs)

    def next(self):
        raise NotImplementedError()

    def set_f_energy(self, model, f_energy=None, condition=False, **chain_args):

        # Energy function -----------------------------------------------------
        X = T.tensor3('x', dtype=floatX)
        x_p = T.matrix('x_p', dtype=floatX)
        h_p = T.matrix('h_p', dtype=floatX)

        inps = [X, x_p, h_p]

        self.dim_h = model.dim_h

        if condition:
            C = T.matrix('C', dtype=floatX)
            inps.append(C)
            if f_energy is None:
                f_energy = model.step_energy_cond
        else:
            C = None
            if f_energy is None:
                f_energy = model.step_energy

        energy, h_n, p_n = f_energy(*(inps + model.get_sample_params()))
        self.f_energy = theano.function(inps, [energy, p_n, h_n])

        # Chain function ------------------------------------------------------
        P = T.scalar('P', dtype='int64')
        inps = [X, h_p]

        if condition:
            inps.append(C)
        inps.append(P)

        chain_dict, updates = model.assign(X, h_p, condition_on=C, **chain_args)
        chain = chain_dict['chain']
        counts = chain_dict['counts']
        scalings = chain_dict['scalings']

        chain += P
        perc = counts.sum() / counts.shape[0].astype(floatX)
        self.f_chain = theano.function(inps, [chain, perc], updates=updates)

    def _build_chain_py(self, x, data_pos):
        h_p = np.random.normal(
            loc=0, scale=1, size=(self.dim_h,)).astype(floatX)

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

    def _build_chain(self, trim_end=0, condition_on=None):
        self.chain = []
        n_remaining_samples = self.dataset.n - self.dataset.pos
        l_chain = min(self.chain_length, n_remaining_samples)

        data_pos = self.dataset.pos
        x, _ = self.dataset.next(batch_size=l_chain)
        n_samples = min(self.build_batch, l_chain)

        t0 = time.time()
        if self.use_theano:
            print('Resetting chain (%s). Position in data is %d'
                  % (self.dataset.mode, data_pos))
            x = x[:, None, :]
            h0 = self.rng.normal(loc=0., scale=1., size=(1, self.dim_h,)).astype('float32')
            inps = [x, h0]
            if condition_on is not None:
                inps.append(condition_on)
            inps.append(data_pos)
            self.chain, perc = self.f_chain(*inps)
            self.chain = self.chain[:, 0].astype('int64')
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
                      self.dim)
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

    def _next(self, l_chain=None, condition_on=None):
        assert self.f_energy is not None

        if self.cpos == -1:
            self.cpos = 0
            self._build_chain(trim_end=self.trim_end, condition_on=condition_on)
            window = min(self.window, len(self.chain))
            self.chain_idx = range(0, len(self.chain) - window + 1, self.chain_stride)
            random.shuffle(self.chain_idx)

        window = min(self.window, len(self.chain))

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