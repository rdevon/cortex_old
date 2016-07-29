'''Setup scripts for Cortex.

'''
import logging
import readline, glob
from os import path
import urllib2

from .manager import get_manager
from .utils.tools import get_paths, _p
from .utils.extra import (
    complete_path, query_yes_no, write_default_theanorc, write_path_conf)


__version__ = '0.3a'
logger = logging.getLogger(__name__)
manager = get_manager()

def resolve_class(cell_type, classes=None):
    from .models import _classes
    if classes is None:
        classes = _classes
    try:
        C = classes[cell_type]
    except KeyError:
        raise KeyError('Unexpected cell subclass `%s`, '
                       'available classes: %s' % (cell_type, classes.keys()))
    return C

def main():
    from cortex.datasets import fetch_basic_data
    from cortex.datasets.neuroimaging import fetch_neuroimaging_data

    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete_path)
    print ('Welcome to Cortex: a deep learning toolbox for '
            'neuroimaging')
    print ('Cortex requires that you enter some paths for '
            'default dataset and output directories. These '
            'can be changed at any time and are customizable '
            'via the ~/.cortexrc file.')

    try:
        path_dict = get_paths()
    except ValueError:
        path_dict = dict()

    if '$data' in path_dict:
        data_path = raw_input(
            'Default data path: [%s] ' % path_dict['$data']) or path_dict['$data']
    else:
        data_path = raw_input('Default data path: ')
    data_path = path.expanduser(data_path)
    if not path.isdir(data_path):
        raise ValueError('path %s does not exist. Please create it.' % data_path)

    if '$outs' in path_dict:
        out_path = raw_input(
            'Default output path: [%s] ' % path_dict['$outs']) or path_dict['$outs']
    else:
        out_path = raw_input('Default output path: ')
    out_path = path.expanduser(out_path)
    if not path.isdir(out_path):
        raise ValueError('path %s does not exist. Please create it.' % out_path)
    write_path_conf(data_path, out_path)

    print ('Cortex demos require additional data that is not necessary for '
           'general use of the Cortex as a package.'
           'This includes MNIST, Caltech Silhoettes, and some UCI dataset '
           'samples.')

    answer = query_yes_no('Download basic dataset? ')

    if answer:
        try:
            fetch_basic_data()
        except urllib2.HTTPError:
            print 'Error: basic dataset not found.'

    print ('Cortex also requires neuroimaging data for the neuroimaging data '
           'for the neuroimaging demos. These are large and can be skipped.')

    answer = query_yes_no('Download neuroimaging dataset? ')

    if answer:
        try:
            fetch_neuroimaging_data()
        except urllib2.HTTPError:
            print 'Error: neuroimaging dataset not found.'

    home = path.expanduser('~')
    trc = path.join(home, '.theanorc')
    if not path.isfile(trc):
        print 'No %s found, adding' % trc
        write_default_theanorc()