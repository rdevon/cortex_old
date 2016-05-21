'''For command-line scripts of demos

'''

from os import path
import sys

from ..utils.training import set_experiment


packagedir = __path__[0]
d = path.join(path.dirname(packagedir), '..')

def run_demo(yaml_file, train):
    args = dict(experiment=yaml_file)

    exp_dict = set_experiment(args)
    train(**exp_dict)

def run_classifier_demo():
    from demos_basic.classifier import train

    yaml_file = path.join(d, 'classifier_mnist.yaml')
    run_demo(yaml_file, train)

def run_rbm_demo():
    from demos_basic.rbm_mnist import train

    yaml_file = path.join(d, 'rbm_mnist.yaml')
    run_demo(yaml_file, train)

def run_vae_demo():
    from demos_basic.vae import train

    yaml_file = path.join(d, 'vae_mnist.yaml')
    run_demo(yaml_file, train)