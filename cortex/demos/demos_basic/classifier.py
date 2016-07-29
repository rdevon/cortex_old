'''Basic classifier script demo.

'''

import cortex
print cortex.__file__
print cortex.__version__
from cortex.utils.tools import print_section
from cortex.utils import floatX, logger as cortex_logger


cortex_logger.set_stream_logger(2)

manager = cortex.get_manager()

print_section('Setting up data')
manager.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
manager.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

print_section('Forming model')
manager.prepare_cell('DistributionMLP', name='mlp', dim_hs=[200, 100],
                     weight_noise=0.01, dropout=0.5)

manager.add_step('mlp', 'mnist.input')
manager.match_dims('mlp.P', 'mnist.labels')
manager.build()

print_section('Adding costs / stats')
manager.add_cost('mlp.negative_log_likelihood', 'mnist.labels')
manager.add_stat('logistic_regression', P='mlp.P', Y_hat='mnist.labels')

print_section('Setting up trainer and evaluator')
train_session = manager.create_session()
manager.build_session()

trainer = manager.setup_trainer(
    manager.get_session(),
    optimizer='sgd',
    epochs=1000,
    learning_rate=0.1,
    batch_size=100
)

valid_session = manager.create_session(noise=False)
manager.build_session()

evaluator = manager.setup_evaluator(
    valid_session,
    valid_stat='logistic_regression.error'
)

print_section('Setting up monitor')
monitor = manager.setup_monitor(modes=['train', 'valid'])
monitor.add_section('cost', keys=['total_cost']+valid_session.costs.keys())
monitor.add_section('stats', keys=valid_session.stats.keys())

print_section('Training')
manager.train(['train', 'valid'], 'valid')