'''
RBM demo with MNIST.
'''

import cortex
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)

batch_size=100
learning_rate=0.01
n_chains=10
n_steps=1
dim_h = 200
epochs = 1000

cortex.set_path('demo_rbm')

cortex.prepare_data('MNIST', mode='train', name='data', source='$data/basic/mnist.pkl.gz')
cortex.prepare_data('MNIST', mode='valid', name='data', source='$data/basic/mnist.pkl.gz')

cortex.prepare_cell('RBM', name='rbm', dim_h=dim_h, h_dist_type='binomial',
                    n_persistent_chains=n_chains)
cortex.match_dims('rbm.input', 'data.input')
cortex.add_step('rbm', 'data.input', n_steps=n_steps, persistent=True)
f = lambda x: x.T
cortex.add_step(f, 'rbm.W', name='W_T')
#cortex.add_step('rbm.update_partition', k=1000)

cortex.build()
cortex.profile()

cortex.add_cost('rbm.cost', X='data.input', V0='rbm.V0', Vk='rbm.Vk')

train_session = cortex.create_session()
cortex.build_session(test=False)

trainer = cortex.setup_trainer(
    train_session, optimizer='sgd', epochs=epochs, learning_rate=learning_rate,
    batch_size=batch_size)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

visualizer = cortex.setup_visualizer(
    valid_session,
    batch_size=10)
visualizer.add('data.viz',
               X='W_T.output',
               name='rbm_weights')

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='rbm.cost.nll',
    batch_size=batch_size
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

cortex.train(eval_every=1, extra_update=cortex._manager.cells['rbm'].update_partition())