'''
Iterative refinement of the approximate posterior
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from ..models import Cell
from ..utils import concatenate, floatX, scan
from ..utils.tools import update_dict_of_lists


class IRVI(Cell):
    '''Iterative refinement of the approximate posterior.

    Will take a variety of methods as the refinement step (see GDIR and AIR).

    NOTE: this class will *not* perform inference by itself, but needs to be
    instantiated from one of the child classes.

    '''
    _call_args = ['Y', 'Q0']
    _args = ['pass_gradients']
    _sample_tensors = ['Qk']

    def __init__(self, models=None, name='IRVI', pass_gradients=False,
                 **kwargs):
        '''Initialization function for IRVI.

        Args:

        '''
        self.pass_gradients = pass_gradients

        super(IRVI, self).__init__(name=name, models=models, **kwargs)

    def set_components(self, models=None, **kwargs):
        self.component_keys = models.keys()
        for k, v in models.iteritems():
            if (not v in self.manager.cells.keys() and
                v in self.manager.cell_args.keys()):
                self.manager.build_cell(v)
            elif v not in self.manager.cell_args.keys():
                raise ValueError('Cell `%s` not foud.' % v)
            self.__dict__[k] = self.manager[v]
        return kwargs

    # Child-specific methods. These must be defined in child class.
    def step_infer(self, *params):
        '''Step inference for `scan`.

        Args:
            *params: shared parameters.

        '''
        raise NotImplementedError()

    def init_infer(self, q):
        '''Initialize inference.

        Args:
            q (T.tensor)

        '''
        raise NotImplementedError()

    def unpack_infer(self, outs):
        '''Unpack inference.

        Args:
            outs (list)

        '''
        raise NotImplementedError()

    def params_infer(self):
        '''Parameters for inference.

        '''
        raise NotImplementedError()

    def init_args(self, Y, Q0, n_samples=None, n_steps=None, inference_rate=None):
        if (n_samples is None or n_steps is None):
            raise TypeError
        return (Y, Q0, n_samples, n_steps, inference_rate)

    def _feed(self, Y, Q0, n_samples, n_steps, inference_rate, *params):
        '''Perform inference

        Args:
            x (T.tensor): Input data sample for posterior, p(h|x)
            y (T.tensor): Output data sample for conditional, p(x|h)
            q0 (Optional[T.tensor]): Initial posterior parameters.

        Returns:
            OrderedDict: Results from inference.
            list: constants in learning.
            theano.OrderedUpdates: updates.

        '''
        Q0 = Q0.copy()
        updates = theano.OrderedUpdates()

        # Set random variables.
        if n_steps > 0:
            epsilons = self.generate_random_variables((n_steps, n_samples), P=Q0)
        else:
            epsilons = self.generate_random_variables((n_samples,), P=Q0)
        rval = OrderedDict()
        rval['Q0'] = Q0
        rval['epsilons'] = epsilons

        # Set `scan` arguments.
        seqs = [epsilons]
        outputs_info = [Q0] + self.init_infer(Q0) + [None, None]
        non_seqs = [Y] + self.params_infer(inference_rate) + list(params)

        self.logger.info('Doing %d inference steps of %s and a rate of %.5f with %d '
               'inference samples' % (n_steps, self.name,
                                      inference_rate, n_samples))

        # Perform inference.
        if n_steps > 1:
            outs, updates_i = scan(self.step_infer, seqs, outputs_info, non_seqs,
                                   n_steps, self.name + '_infer')
            updates.update(updates_i)
            Qs, i_costs, extras = self.unpack_infer(outs)
            Qs_ = Qs
            extra = extras[0]
            Qs = T.concatenate([Q0[None, :, :], Qs], axis=0)
        elif n_steps == 1:
            inps = [epsilons[0]] + outputs_info[:-2] + non_seqs
            outs = self.step_infer(*inps)
            Q, i_cost, extra = self.unpack_infer(outs)
            Qs_ = Q
            Qs = T.concatenate([Q0[None, :, :], Q[None, :, :]], axis=0)
            i_costs = [i_cost]
        elif n_steps == 0:
            Qs = Q0[None, :, :]
            Qs_ = Q0.copy()
            extra = Q0
            i_costs = [T.constant(0.).astype(floatX)]

        rval['extra'] = extra
        rval['Qk'] = Qs[-1]
        rval['Qs'] = Qs
        rval['i_costs'] = i_costs

        if self.pass_gradients:
            constants = []
        else:
            constants = [Qs_]

        rval.update(constants=constants, updates=updates)
        return rval

    def _stats(self, Qs=None, i_costs=None, n_steps=None, epsilons=None,
               isteps=4, **kwargs):
        rval = OrderedDict()
        rval['_delta_Q_mean'] = (Qs[-1] - Qs[0]).mean()
        rval['_delta_i_cost'] = i_costs[-1] - i_costs[0]

        if isteps > 4:
            isteps = range(0, isteps, isteps // 4 + 1)
        else:
            isteps = range(isteps)

        for i in isteps:
            try:
                rval['i_cost({0:02d})'.format(i)] = i_costs[i]
            except:
                pass
        return rval


class DeepIRVI(IRVI):
    '''Deep iterative refinement of the approximate posterior.

    Will take a variety of methods as the refinement step (see DeepGDIR and DeepAIR).
    NOTE: this class will *not* perform inference by itself, but needs to be
    instantiated from one of the child classes.

    '''

    _call_args = ['Y', 'Q0']
    _args = ['pass_gradients']
    _sample_tensors = ['Qk']

    def __init__(self, name='IRVI', pass_gradients=False, **kwargs):
        '''Initialization function for DeepIRVI.

        Args:

        '''

        super(DeepIRVI, self).__init__(name=name, pass_gradients=pass_gradients,
                                       **kwargs)

    def set_components(self, models=None, **kwargs):
        self.component_keys = models.keys()

        def add_component(comp):
            if (not comp in self.manager.cells.keys() and
                comp in self.manager.cell_args.keys()):
                self.manager.build_cell(comp)
            elif comp not in self.manager.cell_args.keys():
                raise ValueError('Cell `%s` not foud.' % comp)

        for k, v in models.iteritems():
            if isinstance(v, (list, tuple)):
                self.__dict__[k] = []
                for v_ in v:
                    add_component(v_)
                    self.__dict__[k].append(self.manager[v_])
            else:
                add_component(v)
                self.__dict__[k] = self.manager[v]
        return kwargs

    def prepare_Qs(self, *Q0s):
        return concatenate(list(Q0s), axis=Q0s[0].ndim-1)

    def init_args(self, Y, Q0, n_samples=None, n_steps=None, inference_rate=None):
        if (n_samples is None or n_steps is None):
            raise TypeError
        return (Y, Q0, n_samples, n_steps, inference_rate)

    def _feed(self, Y, Q0, n_samples, n_steps, inference_rate, *params):
        '''Perform inference

        Args:
            x (T.tensor): Input data sample for posterior, p(h|x)
            y (T.tensor): Output data sample for conditional, p(x|h)
            q0 (Optional[T.tensor]): Initial posterior parameters.

        Returns:
            OrderedDict: Results from inference.
            list: constants in learning.
            theano.OrderedUpdates: updates.

        '''
        updates = theano.OrderedUpdates()
        batch_size = session.batch_size
        if batch_size is not None:
            session.batch_size = n_samples * batch_size

        # Set random variables.
        if n_steps > 0:
            epsilons = self.generate_random_variables((n_steps, n_samples), P=Q0)
        else:
            epsilons = self.generate_random_variables((n_samples,), P=Q0)

        rval = OrderedDict()
        rval['Q0'] = Q0
        rval['epsilons'] = epsilons

        # Set `scan` arguments.
        seqs = [epsilons]
        outputs_info = [Q0] + self.init_infer(Q0) + [None, None]
        non_seqs = [Y] + self.params_infer(inference_rate) + list(params)

        self.logger.info('Doing %d inference steps of %s and a rate of %.5f with %d '
               'inference samples' % (n_steps, self.name,
                                      inference_rate, n_samples))

        # Perform inference.
        if n_steps > 1:
            outs, updates_i = scan(self.step_infer, seqs, outputs_info, non_seqs,
                                   n_steps, self.name + '_infer')
            updates.update(updates_i)
            Qs, i_costs, extras = self.unpack_infer(outs)
            extra = extras
            Qs = T.concatenate([Q0[None, :, :], Qs], axis=0)
            Qk = Qs[-1]
            Qs_ = Qk
        elif n_steps == 1:
            inps = [epsilons[0]] + outputs_info[:-2] + non_seqs
            outs = self.step_infer(*inps)
            Q, i_cost, extra = self.unpack_infer(outs)
            Qs_ = Q[None, :, :]
            Qs = T.concatenate([Q0[None, :, :], Qs_], axis=0)
            i_costs = [i_cost]
            Qk = Q
        elif n_steps == 0:
            Qs = Q0[None, :, :]
            Qs_ = Qs.copy()
            Qk = Q0
            extra = Qs
            i_costs = [T.constant(0.).astype(floatX)]

        rval['extra'] = extra
        rval['Qk'] = Qk
        rval['Qs'] = Qs
        rval['i_costs'] = i_costs
        rval.update(updates=updates)
        rval.update(constants=[Qs_])

        session.batch_size = batch_size
        return rval
