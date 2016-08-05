'''Test module.

'''

import cortex


def test_noise():
    cortex.reset()
    data = cortex.prepare_data('dummy', name='data', n_samples=103, data_shape=(13,))
    noise = cortex.prepare_samples('binomial')
    output = cortex.prod(data, noise)
