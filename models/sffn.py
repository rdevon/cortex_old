'''
Module of Stochastic Feed Forward Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from layers import MLP
import tools
from tools import init_rngs
from tools import init_weights
from tools import log_mean_exp


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX


class SFFN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 cond_to_h=None, cond_from_h=None,
                 weight_scale=1.0, weight_noise=False,
                 z_init=None, learn_z=False,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 momentum=0.9, b1=0.9, b2=0.999,
                 inference_rate=0.1, inference_decay=0.99, n_inference_steps=30,
                 inference_step_scheduler=None,
                 inference_method='sgd', name='sffn'):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.cond_to_h = cond_to_h
        self.cond_from_h = cond_from_h

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale

        self.momentum = momentum
        self.b1 = b1
        self.b2 = b2

        self.z_init = z_init
        self.learn_z = learn_z

        self.x_mode = x_noise_mode
        self.y_mode = y_noise_mode
        self.noise_amount = noise_amount

        self.inference_rate = inference_rate
        self.inference_decay = inference_decay

        self.n_inference_steps = T.constant(n_inference_steps).astype('int64')
        self.inference_step_scheduler = inference_step_scheduler

        if inference_method == 'sgd':
            self.step_infer = self._step_sgd
            self.init_infer = self._init_sgd
            self.unpack_infer = self._unpack_sgd
            self.params_infer = self._params_sgd
        elif inference_method == 'momentum':
            self.step_infer = self._step_momentum
            self.init_infer = self._init_momentum
            self.unpack_infer = self._unpack_momentum
            self.params_infer = self._params_momentum
        elif inference_method == 'adam':
            self.step_infer = self._step_adam
            self.init_infer = self._init_adam
            self.unpack_infer = self._unpack_adam
            self.params_infer = self._params_adam
        elif inference_method == 'cg':
            self.step_infer = self._step_cg
            self.init_infer = self._init_cg
            self.unpack_infer = self._unpack_cg
            self.params_infer = self._params_cg
        elif inference_method == 'cg2':
            self.step_infer = self._step_cg2
            self.init_infer = self._init_cg2
            self.unpack_infer = self._unpack_cg2
            self.params_infer = self._params_cg2
        else:
            raise ValueError()

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        super(SFFN, self).__init__(name=name)

    def set_params(self):
        self.params = OrderedDict()
        if self.cond_to_h is None:
            self.cond_to_h = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='T.nnet.sigmoid')
        if self.cond_from_h is None:
            self.cond_from_h = MLP(self.dim_h, self.dim_out, self.dim_out, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.cond_to_h.name = self.name + '_cond_to_h'
        self.cond_from_h.name = self.name + '_cond_from_h'

    def set_tparams(self):
        tparams = super(SFFN, self).set_tparams()
        tparams.update(**self.cond_to_h.set_tparams())
        tparams.update(**self.cond_from_h.set_tparams())

        return tparams

    def init_z(self, x, y):
        z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)
        return z

    def _sample(self, p, size=None):
        if size is None:
            size = p.shape
        return self.trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

    def _noise(self, x, amount, size):
        return x * (1 - self.trng.binomial(p=amount, size=size, n=1,
                                           dtype=x.dtype))

    def set_input(self, x, mode, size=None):
        if size is None:
            size = x.shape
        if mode == 'sample':
            x = self._sample(x[None, :, :], size=size)
        elif mode == 'noise':
            x = self._sample(x)
            x = self._noise(x[None, :, :], size=size)
        elif mode == None:
            x = self._sample(x, size=x.shape)
            x = T.alloc(0., *size) + x[None, :, :]
        else:
            raise ValueError()
        return x

    def init_inputs(self, x, y, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        y_size = (steps, y.shape[0], y.shape[1])

        x = self.set_input(x, self.x_mode, size=x_size)
        y = self.set_input(y, self.y_mode, size=y_size)
        return x, y

    def get_params(self):
        return self.cond_from_h.get_params()

    def p_y_given_h(self, h, *params):
        return self.cond_from_h.step_call(h, *params)

    def sample_energy(self, ph, y, z, n_samples=10):
        mu = T.nnet.sigmoid(z)

        if n_samples == 0:
            h = mu
        else:
            h = self.cond_to_h.sample(mu, size=(n_samples,
                                                mu.shape[0], mu.shape[1]))

        py = self.cond_from_h(h)

        h_energy = self.cond_to_h.neg_log_prob(h, ph[None, :, :])
        h_energy = -log_mean_exp(-h_energy, axis=0).mean()
        y_energy = self.cond_from_h.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()

        return (h_energy, y_energy)

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    def inference_cost(self, ph, y, z, *params):
        mu = T.nnet.sigmoid(z)
        py = self.p_y_given_h(mu, *params)

        cost = (self.cond_from_h.neg_log_prob(y, py)
                + self.cond_to_h.neg_log_prob(mu, ph)
                - self.cond_to_h.entropy(mu)
                ).sum(axis=0)
        grad = theano.grad(cost, wrt=z, consider_constant=[ph, y])
        return cost, grad

    # SGD
    def _step_sgd(self, ph, y, z, l, *params):
        cost, grad = self.inference_cost(ph, y, z, *params)
        z = (z - l * grad).astype(floatX)
        l *= self.inference_decay
        return z, l, cost

    def _init_sgd(self, ph, y, z):
        return [self.inference_rate]

    def _unpack_sgd(self, outs):
        zs, ls, costs = outs
        return zs, costs

    def _params_sgd(self):
        return []

    # Momentum
    def _step_momentum(self, ph, y, z, l, dz_, m, *params):
        cost, grad = self.inference_cost(ph, y, z, *params)
        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        l *= self.inference_decay
        return z, l, dz, cost

    def _init_momentum(self, ph, y, z):
        return [self.inference_rate, T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs = outs
        return zs, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    # Adam
    def _step_adam(self, ph, y, z, m_tm1, v_tm1, cnt, b1, b2, lr, *params):

        b1 = b1 * (1 - 1e-8)**cnt
        cost, grad = self.inference_cost(ph, y, z, *params)
        m_t = b1 * m_tm1 + (1 - b1) * grad
        v_t = b2 * v_tm1 + (1 - b2) * grad**2
        m_t_hat = m_t / (1. - b1**(cnt + 1))
        v_t_hat = v_t / (1. - b2**(cnt + 1))
        grad_t = m_t_hat / (T.sqrt(v_t_hat) + 1e-8)
        z_t = (z - lr * grad_t).astype(floatX)
        cnt += 1

        return z_t, m_t, v_t, cnt, cost

    def _init_adam(self, ph, y, z):
        return [T.zeros_like(z), T.zeros_like(z), 0]

    def _unpack_adam(self, outs):
        zs, ms, vs, cnts, costs = outs
        return zs, costs

    def _params_adam(self):
        return [T.constant(self.b1).astype('float32'),
                T.constant(self.b2).astype('float32'),
                T.constant(self.inference_rate).astype('float32')]

    def _inference_cost_cg(self, ph, y, z, *params):
        mu = T.nnet.sigmoid(z)
        py = self.p_y_given_h(mu, *params)
        cost = (self.cond_from_h.neg_log_prob(y, py)
                + self.cond_to_h.neg_log_prob(mu, ph)
                - self.cond_to_h.entropy(mu)
                )
        return cost

    # Conjugate gradient with log-grid line search
    def _step_cg(self, ph, y, z, s_, dz_sq_, alphas, *params):
        cost, grad = self.inference_cost(ph, y, z, *params)
        dz = -grad
        dz_sq = (dz * dz).sum(axis=1)
        beta = dz_sq / (dz_sq_ + 1e-8)
        s = dz + beta[:, None] * s_
        z_alpha = z[None, :, :] + alphas[:, None, None] * s[None, :, :]
        costs = self._inference_cost_cg(
            ph[None, :, :], y[None, :, :], z_alpha, *params)
        idx = costs.argmin(axis=0)
        z = z + alphas[idx][:, None] * s
        return z, s, dz_sq, cost

    def _init_cg(self, ph, y, z):
        params = self.get_params()
        s0 = T.zeros_like(z)
        dz_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
        return [s0, dz_sq0]

    def _unpack_cg(self, outs):
        zs, ss, dz_sqs, costs = outs
        return zs, costs

    def _params_cg(self, ):
        return [(self.inference_rate * 2. ** T.arange(8)).astype(floatX)]

    # Inference
    def inference(self, x, y, n_samples=100):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=self.n_inference_steps)
        ph = self.cond_to_h(xs)
        z0 = self.init_z(xs[0], ys[0])

        if self.inference_step_scheduler is None:
            n_inference_steps = self.n_inference_steps
        else:
            n_inference_steps, updates_c = self.inference_step_scheduler(n_inference_steps)
            updates.update(updates_c)

        seqs = [ph, ys]
        outputs_info = [z0] + self.init_infer(ph[0], ys[0], z0) + [None]
        non_seqs = self.params_infer() + self.get_params()

        outs, updates_2 = theano.scan(
            self.step_infer,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'infer'),
            n_steps=n_inference_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        zs, i_costs = self.unpack_infer(outs)
        h_energy, y_energy = self.sample_energy(ph[0], ys[0], zs[-1],
                                                n_samples=n_samples)
        return (xs, ys, zs, h_energy, y_energy, i_costs[-1]), updates

    def __call__(self, x, y, ph=None, n_samples=100, from_z=False):
        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        if ph is not None:
            pass
        elif from_z:
            assert self.learn_z
            zh = T.tanh(T.dot(x_n, self.W0) + self.b0)
            z = T.dot(zh, self.W1) + self.b1
            ph = T.nnet.sigmoid(z)
        else:
            ph = self.cond_to_h(x)

        h = self.cond_to_h.sample(ph, size=(n_samples, ph.shape[0], ph.shape[1]))
        py = self.cond_from_h(h)
        y_energy = self.cond_from_h.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()
        return py, y_energy


class SFFN_MultiLayer(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers,
                 layers=None,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 inference_procedure=None,  inference_parameters=False,
                 name='sffn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.n_layers = n_layers
        assert inference_procedure is not None
        inference_procedure.model = self
        self.procedure = inference_procedure
        self.inference_parameters = inference_parameters

        assert n_layers > 1, 'Hwhat???'

        self.layers = layers

        self.x_mode = x_noise_mode
        self.y_mode = y_noise_mode
        self.noise_amount = noise_amount

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        assert len(kwargs) == 0, kwargs.keys()
        super(SFFN_MultiLayer, self).__init__(name=name)

    def set_params(self):
        self.params = OrderedDict()
        if self.layers is None:
            self.layers = [None for _ in xrange(self.n_layers)]
        else:
            assert self.n_layers == len(layers)

        for i in xrange(self.n_layers):
            if i == 0:
                dim_in = self.dim_in
                dim_out = self.dim_h
            elif i == self.n_layers - 1:
                dim_in = self.dim_h
                dim_out = self.dim_out
            else:
                dim_in = self.dim_h
                dim_out = self.dim_h

            layer = self.layers[i]
            if layer is None:
                self.layers[i] = MLP(dim_in, self.dim_h, dim_out, 1,
                                     rng=self.rng, trng=self.trng,
                                     h_act='T.nnet.sigmoid',
                                     out_act='T.nnet.sigmoid',
                                     name='layer_%d' % i)
            else:
                assert layer.dim_in == dim_in
                assert layer.dim_out == dim_out
                layer.name = 'layer_%d' % i

        if self.inference_parameters:
            for i in xrange(self.n_layers - 1):
                if i == 0:
                    dim_in = self.dim_in
                else:
                    dim_in = self.dim_h

                W = tools.norm_weight(dim_in, self.dim_h,
                                      scale=self.weight_scale, ortho=False)
                self.params['WI_%d' % i] = W

    def set_tparams(self):
        tparams = super(SFFN_MultiLayer, self).set_tparams()
        for layer in self.layers:
            tparams.update(**layer.set_tparams())

        return tparams

    def get_params(self):
        params = []
        for layer in self.layers[1:]:
            params += layer.get_params()

        if self.inference_parameters:
            for i in xrange(1, self.n_layers - 1):
                W = self.__dict__['WI_%d' % i]
                params.append(W)

        return params

    def get_layer_args(self, level, *args):
        assert level > 0
        start = sum([0] + [len(layer.get_params())
                           for layer in self.layers[1:level]])
        length = len(self.layers[level].get_params())
        largs = args[start:start+length]
        return largs

    def get_layer_iargs(self, ):
        pass

    def init_z(self, size, name=''):
        z = T.alloc(0., *size).astype(floatX)
        z.name = name
        return z

    def _sample(self, p, size=None):
        if size is None:
            size = p.shape
        return self.trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

    def _noise(self, x, amount, size):
        return x * (1 - self.trng.binomial(p=amount, size=size, n=1,
                                           dtype=x.dtype))

    def set_input(self, x, mode, size=None):
        if size is None:
            size = x.shape
        if mode == 'sample':
            x = self._sample(x[None, :, :], size=size)
        elif mode == 'noise':
            x = self._sample(x)
            x = self._noise(x[None, :, :], size=size)
        elif mode == None:
            x = self._sample(x, size=x.shape)
            x = T.alloc(0., *size) + x[None, :, :]
        else:
            raise ValueError()
        return x

    def init_inputs(self, x, y, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        y_size = (steps, y.shape[0], y.shape[1])

        x = self.set_input(x, self.x_mode, size=x_size)
        y = self.set_input(y, self.y_mode, size=y_size)
        return x, y

    def sample_energy(self, ph1, preact_h1, y, zs, n_samples=10):
        preacts = []
        pys = [ph1[None, :, :]]
        if n_samples == 0:
            ys = [T.nnet.sigmoid(z) for z in zs]
            pys += [layer(mu) for layer, mu in zip(self.layers[1:], ys)]
        else:
            mu = T.nnet.sigmoid(zs[0])
            h = self._sample(p=mu, size=(n_samples, mu.shape[0], mu.shape[1]))
            hs = [h]
            ys = [mu]
            for i, (z, layer) in enumerate(zip(zs[1:], self.layers[1:-1])):
                py = layer(mu)
                pys.append(py)
                if self.inference_parameters:
                    W = self.__dict__['WI_%d' % (i + 1)]
                    preact = T.dot(h, W)
                else:
                    preact = layer(h, return_preact=True)
                preacts.append(preact)
                mu = T.nnet.sigmoid(preact + z[None, :, :])
                h = self._sample(p=mu)
                hs.append(h)
                ys.append(mu)

        pys.append(self.layers[-1](mu))
        ys.append(y[None, :, :])

        energies = []

        for layer, py, y in zip(self.layers, pys, ys):
            energy = layer.neg_log_prob(y, py)#.mean()
            energy = -log_mean_exp(-energy, axis=0).mean()
            energies.append(energy)

        if self.inference_parameters:
            for preact, layer, mu in zip(preacts, self.layers[1:-1], ys[1:-1]):
                energy = layer.neg_log_prob(mu, )
                energy = -log_mean_exp()

        return energies, ys

    def inference_cost(self, *args):
        ph1, preact_h1, y = args[:3]
        zs = self.get_qparams(*args)
        args = args[-len(self.get_params()):]

        mu = T.nnet.sigmoid(zs[0])
        h = self._sample(p=mu)
        py = ph1
        preacts = []

        cost = -self.layers[0].entropy(mu)
        cost += self.layers[0].neg_log_prob(mu, py)
        for l, layer in enumerate(self.layers[1:]):
            params = self.get_layer_args(l + 1, *args)
            py = layer.step_call(mu, *params)

            if l + 1 < self.n_layers - 1:
                if self.inference_parameters:
                    W = self.get_inference_args(l + 1, *args)
                    preact = T.dot(h, W)
                else:
                    preact = layer.preact(mu, *params)
                preacts.append(preact)
                z = zs[l + 1]
                mu = T.nnet.sigmoid(preact + z)
                h = self._sample(p=mu)
                cost += -layer.entropy(mu)
            else:
                mu = y
            cost += layer.neg_log_prob(mu, py)

        cost = cost.sum(axis=0)
        grads = theano.grad(cost, wrt=zs, consider_constant=[])

        return cost, grads

    def get_qparams(self, *args):
        qparams = args[3:3+self.n_layers-1]
        return qparams

    def get_qrest(self, *args):
        qrest = args[3+self.n_layers-1:-len(self.get_params())]
        return qrest

    def inference(self, x, y, z0s=None, n_samples=20):
        n_inference_steps, updates = self.procedure.get_steps()

        xs, ys = self.init_inputs(x, y, steps=n_inference_steps)
        ph1 = self.layers[0](xs)
        preact_h1 = self.layers[0](xs, return_preact=True)

        if z0s is None:
            z0s = [self.init_z((x.shape[0], layer.dim_out), name='z0%d' % i)
                   for i, layer in enumerate(self.layers[:-1])]

        seqs = [ph1, preact_h1, ys]
        outputs_info = z0s + self.procedure.init_inference(xs[0], z0s) + [None]
        non_seqs = self.procedure.get_params() + self.get_params()

        outs, updates_2 = theano.scan(
            self.procedure.step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'infer'),
            n_steps=n_inference_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        zs, i_costs = self.procedure.unpack(outs)
        zs_f = [z[-1] for z in zs]

        energies, hs = self.sample_energy(ph1[0], preact_h1[0], y, zs_f,
                                          n_samples=n_samples)
        return (xs, ys, zs, hs, energies, i_costs[-1]), updates

    def __call__(self, x, y, n_samples=100, from_z=False):
        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
        py = self.layers[0](x_n)
        y_hat = self.layers[0].sample(py, size=(n_samples, py.shape[0], py.shape[1]))

        for layer in self.layers[1:]:
            py = layer(y_hat)
            y_hat = layer.sample(py)

        y_energy = layer.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()
        return py, y_energy
