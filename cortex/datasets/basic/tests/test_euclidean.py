'''Unit test module for Euclidean dataset.

'''

from cortex.datasets.basic.euclidean import Euclidean


def _test_method(method='fibrous'):
    data_iter = Euclidean(metho=method)
    data_iter.next()

def test_methods():
    methods = ['fibrous', 'circle', 'sprial', 'X', 'modes', 'bullseye']
    for method in methods:
        _test_method(method)