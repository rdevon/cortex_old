'''
Testing Dijktras algorithm in Theano
'''

import numpy as np
import theano
from theano import tensor as T

from datasets.euclidean import Euclidean
from datasets.chains import DChains, load_data
from models.chainer import DijkstrasChainer as Chainer
from models.rnn import RNN
from utils import floatX, intX


np.set_printoptions(precision=3)

def test_dijksras():
    E = T.matrix('E', dtype=floatX)
    D = E[:100].T
    X = T.vector('X', dtype=floatX)

    C = T.tile(T.arange(D.shape[0]), D.shape[0])

    def step(i, D, E):
        D_new = T.switch(T.lt(E[i][:, None] + D[i][None, :], D), E[i][:, None] + D[i][None, :], D)
        return D_new, theano.scan_module.until(T.all(T.eq(D, D_new)))

    Ds, updates = theano.scan(
        step,
        sequences=[C],
        outputs_info=[D],
        non_sequences=[E]
    )

    def step_path(i, Q, Ds):
        P = T.switch(T.eq(Ds[i], Ds[i-1]), -1, i % Ds.shape[1])
        Q = T.switch(T.neq(P, -1), P, Q)
        return Q, P

    Q0 = T.zeros_like(D) + T.arange(D.shape[0])[:, None]
    (Qs, Ps), updates = theano.scan(
        step_path,
        sequences=[T.arange(1, Ds.shape[0])[::-1]],
        outputs_info=[Q0, None],
        non_sequences=[Ds]
    )

    idx = T.neq(Ps.sum(axis=(1, 2)), -(D.shape[0] * D.shape[1])).nonzero()[0]
    Qs = Qs[idx]

    Ps = Ps[idx]
    Qs0 = T.zeros_like(D) + T.arange(D.shape[1])[None, :]
    Qsl = T.zeros_like(D) + T.arange(D.shape[0])[:, None]
    Qs = T.concatenate([Qs0[None, :, :], Qs[::-1], Qsl[None, :, :]]).astype(intX)
    Ps = T.concatenate([Qs0[None, :, :], Ps[::-1], Qsl[None, :, :]])

    chain = X[Qs]
    chain = T.switch(T.eq(Ps, -1), 0, chain)
    mask = T.neq(Ps, -1).astype(intX)

    f = theano.function([E, X], [idx, Ds, Qs, chain, mask], on_unused_input='ignore')
    keys = ['idx', 'Ds', 'Qs', 'chain', 'mask']

    N = 500
    edges = np.random.randint(1, 30, size=(N, N)).astype(floatX)
    edges *= (np.ones_like(edges) - np.eye(edges.shape[0]))
    x = np.random.normal(size=(edges.shape[0])).astype(floatX)

    print edges
    #print x
    for k, res in zip(keys, f(edges, x)):
        print k, res, res.shape

    assert False


def test_chainer(dim_in=5, n_samples=53, batch_size=17, data_batch_size=37):
    model = RNN(dim_in, [13])
    model.set_tparams()
    chainer = Chainer(model)

    chain_iter = DChains(Euclidean, model.dim_hs,
                         batch_size=batch_size,
                         data_batch_size=data_batch_size,
                         n_samples=n_samples, dims=dim_in,
                         build_batch=10)
    chain_iter.set_chainer(chainer)

    outs = chain_iter.next()

    x = outs['x']
    hs = outs['hs']
    mask = outs['mask']
    idx = outs['idx']
    data_idx = outs['data_idx']
    print data_idx
    print idx

    idx = idx.reshape((idx.shape[0] * idx.shape[1],))

    X = T.tensor3('X', dtype=floatX)
    H0s = [T.matrix('H%d' % l, dtype=floatX) for l in range(model.n_layers)]
    M = T.matrix('mask', dtype=floatX)
    I = T.vector('idx', dtype=intX)
    DI = T.vector('data_idx', dtype=intX)
    rval, updates = model(X, m=M, h0s=H0s)

    Hs = rval['hs']
    idx_os = T.arange(DI.shape[0])
    H_es = [H.reshape((H.shape[0] * H.shape[1], H.shape[2])) for H in Hs]
    M_e = M.reshape((M.shape[0] * M.shape[1],))

    def step_count(idx_o, I, H, m):
        count = T.switch(T.eq(I, idx_o), 1, 0).astype(floatX) * m
        H_sum = (count[:, None] * H).sum(axis=0).astype(floatX)
        s = T.maximum(count.sum(axis=0), 1).astype(floatX)
        return H_sum / s

    H_us = []
    H_ns = []
    for H_p, H in zip(chain_iter.dataset.Hs, H_es):
        H_n, _ = theano.scan(
            step_count,
            sequences=[idx_os],
            outputs_info=[None],
            non_sequences=[I, H, M_e]
        )
        H_ns.append(H_n)
        H_u = H_p.copy()
        H_u = T.set_subtensor(H_u[DI], H_n)
        H_us.append(H_u)

    updates = [(H_p, H_n) for H_p, H_n in zip(chain_iter.dataset.Hs, H_us)]

    f = theano.function([X] + H0s + [M, I, DI], H_ns, updates=updates)
    print chain_iter.dataset.Hs[0].get_value(), chain_iter.dataset.Hs[0].get_value().shape
    inps = [x] + [h[0] for h in hs] + [mask, idx, data_idx]
    f(*inps)
    print chain_iter.dataset.Hs[0].get_value(), chain_iter.dataset.Hs[0].get_value().shape

    '''
    for k, v in outs.iteritems():
        if isinstance(v, list):
            for v_ in v:
                print k, v_.shape
        else:
            print k, v.shape
    '''

    assert False