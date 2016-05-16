'''
Iterative refinement of the approximate posterior
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from ..utils import floatX
from ..utils.tools import (
    scan,
    update_dict_of_lists,
    warn_kwargs
)

class IRVI(object):
    '''Iterative refinement of the approximate posterior.
    
    Will take a variety of methods as the refinement step (see GDIR and AIR).
    NOTE: this class will *not* perform inference by itself, but needs to be
    instantiated from one of the child classes.
    
    Atrributes:
        name: str. Name of inference method.
        model: Layer. Typically Helmholtz
        init_inference: str. Inference initialization option.
        inference_rate: float. Rate of inference steps.
        n_inference_steps: int. Number of inference steps.
        n_inference_samples: int. Number of samples to draw from the approximate
            posterior.
        pass_gradients: bool. Pass gradients during inference.
        use_all_samples: bool. Use all the samples rather than just last.
    
    '''
    def __init__(self,
                 model,
                 name='IRVI',
                 inference_rate=0.1,
                 n_inference_samples=20,
                 n_inference_steps=20,
                 pass_gradients=True,
                 init_inference='recognition_network',
                 use_all_samples=False,
                 **kwargs):
        '''Initialization function for IRVI.
        
        Args:
            model: Layer object. Typically Helmholtz.
            init_inference: str. Inference initialization option.
            inference_rate: float. Rate of inference steps.
            n_inference_steps: int. Number of inference steps.
            n_inference_samples: int. Number of samples to draw from the approximate
                posterior.
            pass_gradients: bool. Pass gradients during inference.
            use_all_samples: bool. Use all the samples rather than just last.        
        '''

        self.name = name
        self.model = model
        self.init_inference = init_inference
        self.inference_rate = inference_rate
        self.n_inference_steps = n_inference_steps
        self.n_inference_samples = n_inference_samples
        self.pass_gradients = pass_gradients
        self.use_all_samples = use_all_samples

        warn_kwargs(self, **kwargs)

    # Child-specific methods. These must be defined in child class.
    def step_infer(self, *params):  raise NotImplementedError()
    def init_infer(self, q):        raise NotImplementedError()
    def unpack_infer(self, outs):   raise NotImplementedError()
    def params_infer(self):         raise NotImplementedError()

    def init_variational_inference(self, x):
        '''Initialize variational inference.
        
        Args:
            x: T.tensor. Data samples
            
        Returns:
            q0: T.tensor. Initial variational parameters.
        '''
        model = self.model

        if self.init_inference == 'recognition_network':
            print 'Initializing %s inference with recognition network' % self.name
            q0 = model.posterior.feed(x)
        elif self.init_inference == 'from_prior':
            print 'Initializing %s inference with prior parameters' % self.name
            q0 = model.prior.get_center(**model.prior.get_params())
        else:
            raise ValueError(self.init_inference)

        return q0

    def inference(self, x, y, q0=None):
        '''Perform inference
        
        Args:
            x: T.tensor. Input data sample for posterior, p(h|x)
            y: T.tensor. Output data sample for conditional, p(x|h)
            q0: T.tensor (optional). Initial posterior parameters.
        
        Returns:
            rval: OrderedDict. Results from inference.
            constants: list
            updates: OrderedUpdates from scan.
        '''
        model = self.model
        updates = theano.OrderedUpdates()

        # Initialize inference.
        if q0 is None:
            q0 = self.init_variational_inference(x)

        # Set random variables.
        epsilons = model.init_inference_samples(
            size=(self.n_inference_steps, self.n_inference_samples,
                  x.shape[0], model.dim_h))

        # Set `scan` arguments.
        seqs = [epsilons]
        outputs_info = [q0] + self.init_infer(q0) + [None]
        non_seqs = [y] + self.params_infer() + model.get_params()

        print ('Doing %d inference steps of %s and a rate of %.5f with %d '
               'inference samples'
               % (self.n_inference_steps, self.name,
                  self.inference_rate, self.n_inference_samples))

        # Perform inference.
        if self.n_inference_steps > 1:
            print 'Multiple inference steps. Using `scan`'
            outs, updates_i = scan(
                self.step_infer, seqs, outputs_info, non_seqs, self.n_inference_steps,
                self.name + '_infer'
            )
            updates.update(updates_i)
            qs, i_costs = self.unpack_infer(outs)
            qs = T.concatenate([q0[None, :, :], qs], axis=0)
        elif self.n_inference_steps == 1:
            print 'Single inference step'
            inps = [epsilons[0]] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            q, i_cost = self.unpack_infer(outs)
            qs = T.concatenate([q0[None, :, :], q[None, :, :]], axis=0)
            i_costs = [i_cost]
        elif self.n_inference_steps == 0:
            print 'No inference steps'
            qs = q0[None, :, :]
            i_costs = [T.constant(0.).astype(floatX)]

        if self.pass_gradients:
            constants = []
        else:
            constants = [qs]

        if self.use_all_samples:
            qk = qs
        else:
            qk = qs[-1]
        
        rval = OrderedDict(
            qk=qk,
            qs=qs,
            i_costs=i_costs
        )

        return rval, constants, updates

    def test(self, x, y, stride=1, **model_args):
        '''Testing function for inference.
        
        Returns a larger summary across different number of inference steps.
        
        Args:
            x: T.tensor. Input data sample for posterior, p(h|x)
            y: T.tensor. Output data sample for conditional, p(x|h)
            stride: int. Stride for result summary
            model_args: dict. dictionary of arguments for model results.
            
        Returns:
            results: OrderedDict of results. Only first- and last-step results.
            samples: OrderedDict of tensors for visualization, etc.
            full_results: OrderedDict of complete result summary
            updates: OrderedUpdates from scan.
        '''
        model = self.model

        # Perform inference
        inference_outs, _, updates = self.inference(x, y)
        i_costs = inference_outs['i_costs']

        qs = inference_outs['qs']

        # Set up summary steps.
        if self.n_inference_steps > stride and stride != 0:
            steps = [0, 1] + range(stride, self.n_inference_steps, stride)
            steps = steps[:-1] + [self.n_inference_steps - 1]
        elif self.n_inference_steps > 0:
            steps = [0, self.n_inference_steps - 1]
        else:
            steps = [0]

        # Extract results from model
        full_results = OrderedDict()
        full_results['i_cost'] = []
        samples = OrderedDict()
        for i in steps:
            qk  = qs[i]
            results_k, samples_k, _, _ = model(x, y, qk, **model_args)
            samples_k['q'] = qk
            update_dict_of_lists(full_results, **results_k)
            full_results['i_cost'].append(i_costs[i])
            update_dict_of_lists(samples, **samples_k)

        # Final results are from first and last steps
        results = OrderedDict()
        for k, v in full_results.iteritems():
            results[k] = v[-1]
            results[k + '0'] = v[0]
            try:
                results['d_' + k] = v[0] - v[-1]
            except:
                print k, v[0], v[-1]

        return results, samples, full_results, updates

    def __call__(self, x, y, **model_args):
        '''Call function for performing inference.
        
        Args:
            x: T.tensor. Input data sample for posterior, p(h|x)
            y: T.tensor. Output data sample for conditional, p(x|h)
            model_args: dict. dictionary of arguments for model results.
            
        Returns:
            results: OrderedDict
            samples: OrderedDict of tensors
            constants: list
            updates: OrderedUpdates from scan.
        '''
        model = self.model
        inference_outs, constants, updates = self.inference(x, y)
        qk = inference_outs['qk']
        results, samples, constants_m, updates_m = model(x, y, qk=qk, **model_args)
        constants += constants_m
        updates += updates_m
        return results, samples, constants, updates


class DeepIRVI(object):

    def __init__(self,
                 model,
                 name='IRVI',
                 inference_rate=0.1,
                 n_inference_samples=20,
                 n_inference_steps=20,
                 pass_gradients=True,
                 sample_posterior=False,
                 init_inference='recognition_network',
                 **kwargs):

        self.name = name
        self.model = model
        self.init_inference = init_inference
        self.inference_rate = inference_rate
        self.n_inference_steps = n_inference_steps
        self.n_inference_samples = n_inference_samples
        self.pass_gradients = pass_gradients
        self.sample_posterior = sample_posterior
        warn_kwargs(self, **kwargs)

    def step_infer(self, *params):  raise NotImplementedError()
    def init_infer(self, q0s):      raise NotImplementedError()
    def unpack_infer(self, outs):   raise NotImplementedError()
    def params_infer(self):         raise NotImplementedError()

    def init_variational_inference(self, x):
        model = self.model

        if self.init_inference == 'recognition_network':
            print 'Initializing %s inference with recognition network' % self.name
            q0s = []
            state = x[None, :, :]

            for l in xrange(model.n_layers):
                q0 = model.posteriors[l].feed(state).mean(axis=0)
                q0s.append(q0)
                if self.sample_posterior:
                    state, _ = model.posteriors[l].sample(q0, n_samples=n_samples)
                else:
                    state = q0[None, :, :]
        else:
            raise ValueError(self.init_inference)

        return q0s

    def inference(self, x, y, q0s=None):
        model = self.model
        updates = theano.OrderedUpdates()

        if q0s is None:
            q0s = self.init_variational_inference(x)

        epsilons = []

        epsilons = [model.init_inference_samples(
            l, size=(self.n_inference_steps, self.n_inference_samples,
                     x.shape[0], model.dim_hs[l])) for l in xrange(model.n_layers)]

        seqs = epsilons
        outputs_info = q0s + self.init_infer(q0s) + [None]
        non_seqs = [y] + self.params_infer() + model.get_params()

        print ('Doing %d inference steps of %s and a rate of %.5f with %d '
               'inference samples'
               % (self.n_inference_steps, self.name,
                  self.inference_rate, self.n_inference_samples))

        if self.n_inference_steps > 1:
            print 'Multiple inference steps. Using `scan`'
            outs, updates_i = scan(
                self.step_infer, seqs, outputs_info, non_seqs, self.n_inference_steps,
                self.name + '_infer'
            )
            updates.update(updates_i)
            qss, i_costs = self.unpack_infer(outs)
            qss = [T.concatenate([q0[None, :, :], qs], axis=0)
                   for q0, qs in zip(q0s, qss)]

        elif self.n_inference_steps == 1:
            print 'Single inference step'
            inps = [epsilon[0] for epsilon in epsilons] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            qs, i_cost = self.unpack_infer(outs)
            qss = [T.concatenate([q0s[None, :, :], qs[None, :, :]], axis=0)]
            i_costs = [i_cost]

        elif self.n_inference_steps == 0:
            print 'No inference steps'
            qss = [q0[None, :, :] for q0 in q0s]
            i_costs = [T.constant(0.).astype(floatX)]

        if self.pass_gradients:
            constants = []
        else:
            constants = qss

        rval = OrderedDict(
            qks=[qs[-1] for qs in qss],
            qss=qss,
            i_costs=i_costs
        )

        return rval, constants, updates

    def test(self, x, y, stride=10, **model_args):
        model = self.model

        inference_outs, _, updates = self.inference(x, y)
        i_costs = inference_outs['i_costs']

        qss = inference_outs['qss']

        if self.n_inference_steps > stride and stride != 0:
            steps = [0, 1] + range(stride, self.n_inference_steps, stride)
            steps = steps[:-1] + [self.n_inference_steps - 1]
        elif self.n_inference_steps > 0:
            steps = [0, self.n_inference_steps - 1]
        else:
            steps = [0]

        full_results = OrderedDict()
        full_results['i_cost'] = []
        samples = OrderedDict()
        for i in steps:
            qks  = [qs[i] for qs in qss]
            results_k, samples_k, _, _ = model(x, y, qks, **model_args)
            samples_k['qs'] = qks
            update_dict_of_lists(full_results, **results_k)
            full_results['i_cost'].append(i_costs[i])
            update_dict_of_lists(samples, **samples_k)

        results = OrderedDict()
        for k, v in full_results.iteritems():
            results[k] = v[-1]
            results[k + '0'] = v[0]
            results['d_' + k] = v[0] - v[-1]

        return results, samples, full_results, updates
    
    def __call__(self, x, y, **model_args):
        model = self.model
        inference_outs, constants, updates = self.inference(x, y)
        qks = inference_outs['qks']
        results, samples, constants_m, updates_m = model(x, y, qks=qks, **model_args)
        constants += constants_m
        updates += updates_m
        return results, samples, constants, updates
