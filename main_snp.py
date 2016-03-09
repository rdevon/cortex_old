import argparse
from collections import OrderedDict
from glob import glob
import numpy as np
import os
from os import path
import pprint
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    RotatingMarker,
    SimpleProgress,
    Timer
)
import shutil
import sys
import theano
from theano import tensor as T
import time

from datasets import load_data
from inference import inference_class
from models.distributions import Gaussian
from models.gbn import (
    GBN,
    unpack
)
from utils.monitor import SimpleMonitor
from utils import op
from utils.tools import (
    floatX,
    get_trng,
    itemlist,
    load_experiment,
    print_profile,
    print_section,
    resolve_path,
    update_dict_of_lists
)

#import SNP

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    parser.add_argument('-n', '--name', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))


dataset_args = exp_dict['dataset_args']
learning_args = exp_dict['learning_args']
#inference_args_test = exp_dict['inference_args_test']

train, valid, test = load_data(train_batch_size=learning_args['batch_size'], **dataset_args)

x, y = train.next(10)
print x, y
train.shuffle = False
train.reset()
x, y = train.next(5)
print x, y

train.shuffle = True
train.reset()
x, y = train.next(5)
print x, y
#import ipdb
#ipdb.set_trace()

