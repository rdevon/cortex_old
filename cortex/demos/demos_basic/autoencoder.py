'''Basic autoencoder script demo.

'''

import cortex
from cortex.utils.tools import print_section


cortex.prepare_data('MNIST', mode='train', source='$data/basic/mnist.pkl.gz')
cortex.prepare_data('MNIST', mode='valid', source='$data/basic/mnist.pkl.gz')

print_section('Forming model')
cortex.prepare_cell('MLP', name='encoder', dim_hs=[200, 100], dim_out=50,
                     weight_noise=0.01, dropout=0.5)
cortex.prepare_cell('DistributionMLP', name='decoder', dim_hs=[100, 200],
                     weight_noise=0.01, dropout=0.5)

cortex.add_step('noise.binary', 'mnist.input', name='input_noise', noise=0.5)
cortex.add_step('encoder', 'input_noise.output')
cortex.add_step('decoder', 'encoder.output')
cortex.match_dims('decoder.P', 'mnist.input')

cortex.add_step('visualization.random_set',
                 X_in='input_noise.output',
                 X_out='decoder.P',
                 Y='mnist.labels',
                 n_samples=100)

cortex.build()
cortex.profile()

cortex.add_cost('decoder.negative_log_likelihood', 'mnist.input')
cortex.add_cost('l2_decay', 0.0002, 'encoder.weights')
cortex.add_cost('l2_decay', 0.0002, 'decoder.mlp.weights')

train_session = cortex.create_session()
cortex.build_session()

trainer = cortex.setup_trainer(
    train_session,
    optimizer='sgd',
    epochs=1000,
    learning_rate=0.1,
    batch_size=100
)

valid_session = cortex.create_session(noise=False)
cortex.build_session()

evaluator = cortex.setup_evaluator(
    valid_session,
    valid_stat='decoder.negative_log_likelihood'
)

monitor = cortex.setup_monitor(valid_session, modes=['train', 'valid'])

visualizer = cortex.setup_visualizer(valid_session)
visualizer.add('mnist.autoencoder_visualization',
               inputs='visualization.random_set.outputs',
               out_file='/Users/devon/tmp/autoencoder_test.png')

print_section('Training')
cortex.train()