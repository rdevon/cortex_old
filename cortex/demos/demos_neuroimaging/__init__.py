'''For command-line scripts of demos

'''

from os import path
import sys

from ...utils.training import set_experiment


packagedir = __path__[0]
d = path.join(path.dirname(packagedir), '../..')

def run_demo(yaml_file, train):
    args = dict(experiment=yaml_file)

    exp_dict = set_experiment(args)
    train(**exp_dict)

def run_rbm_vbm_demo():
    from .rbm_ni import train

    yaml_file = path.join(d, 'rbm_vbm.yaml')
    run_demo(yaml_file, train)