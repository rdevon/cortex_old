'''
Tests for RNN
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from datasets.euclidean import Euclidean
from models.rnn import RNN
from utils import floatX


sigmoid = lambda x: (1. / (1 + np.exp(-x))) * 0.9999 + 0.000005

def test_build(dim_in=13, dim_h=17):
    rnn = RNN(dim_in, dim_h)
    rnn.set_tparams()

    return rnn

def test_recurrent(dim_in=13, dim_h=17, n_samples=107, window=7):
    rnn = test_build(dim_in, dim_h)

    data_iter = Euclidean(n_samples=n_samples, dims=dim_in, batch_size=window)
    x = data_iter.next()[data_iter.name]

    test_dict = OrderedDict()

    X = T.matrix('x', dtype=floatX)

    Y = rnn.call_seqs(X, None, *rnn.get_sample_params())[0]
    y = np.dot(x, rnn.input_net.params['W0']) + rnn.input_net.params['b0']
    test_dict['RNN preact from data'] = (X, Y, x, y, theano.OrderedUpdates())

    H0 = T.alloc(0., X.shape[0], rnn.dim_h).astype(floatX)
    H = rnn._step(Y, H0, rnn.Ur)
    h0 = np.zeros((x.shape[0], rnn.dim_h)).astype(floatX)
    h = np.tanh(np.dot(h0, rnn.params['Ur']) + y)
    test_dict['step reccurent'] = (X, H, x, h, theano.OrderedUpdates())

    P = rnn.output_net.feed(H)
    p = sigmoid(np.dot(h, rnn.output_net.params['W0']) + rnn.output_net.params['b0'])
    test_dict['output'] = (X, P, x, p, theano.OrderedUpdates())

    for k, v in test_dict.iteritems():
        print 'Testing %s' % k
        inp, out, inp_np, out_np, updates = v
        f = theano.function([inp], out, updates=updates)
        out_actual = f(inp_np)
        if not np.allclose(out_np, out_actual):
            print 'np', out_np
            print 'theano', out_actual
            assert False
