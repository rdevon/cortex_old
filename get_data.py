#/bin/python
'''
Script to get data.
'''

import os
import urllib

from utils.tools import get_paths


data_path = get_paths()['$irvi_data']
if data_path == '':
    raise ValueError('irvi.conf not set yet. Set data path.')

if os.path.isfile(data_path):
    raise ValueError('Data path specified (%s) is a file...' % data_path)
elif not os.path.isdir(data_path):
    os.mkdir(os.path.abspath(data_path))

data_dict = {
    'mnist': 'http://www.capsec.org/datasets/mnist.pkl.gz',
    'mnist_binarized': 'http://www.capsec.org/datasets/mnist_salakhutdinov.pkl.gz',
    'caltech': 'http://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat'
}

for name, url in data_dict.iteritems():
    print 'Downloading %s' % name
    try:
        f_name = url.split('/')[-1]
        urllib.urlretrieve (url, os.path.join(data_path, f_name))
    except Exception as e:
        print e
