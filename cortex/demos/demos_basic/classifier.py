'''Basic classifier script demo.

'''

from os import path

import cortex
from cortex.utils.tools import print_section
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(1)
out_path = '$outs/classifier_demo'

print_section('Setting up data')
cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

print_section('Forming model') # -----------------------------------------------
cortex.prepare_cell('DistributionMLP', name='classifier', dim_hs=[200, 100],
                     weight_noise=0.01, dropout=0.5, batch_normalization=True)

cortex.add_step('classifier', 'mnist.input')
cortex.match_dims('classifier.P', 'mnist.labels')
cortex.add_step('visualization.classifier.random_set',
                 P='classifier.P', Y='mnist.labels', X='mnist.input',
                 n_samples=100)
cortex.build()

print_section('Adding costs / stats')
cortex.add_cost('classifier.negative_log_likelihood', 'mnist.labels')
cortex.add_cost('l2_decay', 0.0002, 'classifier.mlp.weights')
cortex.add_stat('logistic_regression', P='classifier.P', Y='mnist.labels')

print_section('Setting up trainer and evaluator')
train_session = cortex.create_session()
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    epochs=1000,
    optimizer='sgd',
    learning_rate=0.1,
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
visualizer = cortex.setup_visualizer(valid_session)
visualizer.add('mnist.classification_visualization',
               inputs='visualization.classifier.random_set.outputs',
               out_file=path.join(out_path, 'classifier_test.png'))

print_section('Training')
cortex.train(out_path=out_path)