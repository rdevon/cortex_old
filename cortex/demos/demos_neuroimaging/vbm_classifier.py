'''Basic classifier script demo.

'''

import cortex
from cortex.utils.tools import print_section
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)

print_section('Setting up data')
cortex.prepare_data_split('MRI', name='data', split=[.7, .2, .1],
                          source='$data/neuroimaging/VBM_test/VBM.yaml')

print_section('Forming model') # -----------------------------------------------
cortex.prepare_cell('DistributionMLP', name='classifier', dim_hs=[200, 100],
                     weight_noise=0.01, dropout=0.2, batch_normalization=True)

cortex.add_step('classifier', 'data.input')
cortex.match_dims('classifier.P', 'data.labels')
cortex.add_step('visualization.classifier.random_set',
                 P='classifier.P', Y='data.labels', X='data.input',
                 n_samples=100)
cortex.build()

print_section('Adding costs / stats')
cortex.add_cost('classifier.negative_log_likelihood', 'data.labels')
cortex.add_cost('l2_decay', 0.002, 'classifier.mlp.weights')
cortex.add_stat('logistic_regression', P='classifier.P', Y='data.labels')

print_section('Setting up trainer and evaluator')
train_session = cortex.create_session()
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    epochs=1000,
    optimizer='sgd',
    learning_rate=0.001,
    batch_size=100
)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='logistic_regression.error'
)

print_section('Setting up monitor')
monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

cortex.profile()

print_section('Training')
cortex.train()