'''Conv classifier script demo.

'''

import cortex
from cortex.utils.tools import print_section
from cortex.utils import logger as cortex_logger


cortex_logger.set_stream_logger(2)

print_section('Setting up data')
cortex.prepare_data('CIFAR', mode='train', name='data',
                    source='$data/basic/cifar-10-batches-py/')
cortex.prepare_data('CIFAR', mode='valid', name='data',
                    source='$data/basic/cifar-10-batches-py/')

print_section('Forming model') # -----------------------------------------------
cnn_args = dict(
    cell_type='CNN2D',
    input_shape=cortex._manager.datasets['data']['image_shape'],
    filter_shapes=((5, 5), (3, 3), (2, 2), (1, 1)),
    pool_sizes=((2, 2), (5, 5), (1, 1), (1, 1)), n_filters=[20, 50, 100, 10],
    h_act='softplus'
)

cortex.prepare_cell('DistributionMLP', name='classifier',
                    mlp=cnn_args, dropout=0.2, batch_normalization=True)

cortex.add_step('classifier', 'data.input')
cortex.match_dims('classifier.P', 'data.labels')
cortex.add_step('visualization.classifier.random_set',
                 P='classifier.P', Y='data.labels', X='data.input',
                 n_samples=100)
cortex.build()

print_section('Adding costs / stats')
cortex.add_cost('classifier.negative_log_likelihood', 'data.labels')
cortex.add_stat('logistic_regression', P='classifier.P', Y='data.labels')

print_section('Setting up trainer and evaluator')
train_session = cortex.create_session(batch_size=100)
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    epochs=1000,
    optimizer='rmsprop',
    learning_rate=0.0001
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
visualizer.add('data.classification_visualization',
               inputs='visualization.classifier.random_set.outputs',
               out_file='$outs/conv_classifier_test.png')

print_section('Training')
cortex.train()