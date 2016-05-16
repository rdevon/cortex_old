'''
Setup for hippo
'''

import readline, glob
from setuptools import setup, find_packages
from setuptools.command.install import install
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

def complete(text, state):
    return (glob.glob(text + '*') + [None])[state]

def write_path_conf(data_path, out_path):
    with path.join(here, 'paths.conf') as f:
        f.write('[PATHS]\n')
        f.write('$data: %s\n' % path.abspath(data_path))
        f.write('$outs: %s\n' % path.abspath(out_path))

class cortex_install(install):
    def run(self):
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind('tab: complete')
        readline.set_completer(complete)
        print ('Welcome to cortex: a deep learning toolbox for '
               'neuroimaging')
        print ('cortex requires that you enter some paths for '
               'default dataset and output directories. These '
               'can be changed at any time and are customizable '
               'via the paths.conf file.')
        data_path = raw_input('Default data path: ')
        out_path = raw_input('Default output path: ')
    
cmdclass = {'install': cortex_install}
    
setup(
    cmdclass=cmdclass,
    name='cortex',
    version='0.1a',
    description='cortex: a deep learning toolbox for neuroimaging',
    long_description=long_description,
    url='https://github.com/rdevon/cortex',
    author='Devon Hjelm',
    author_email='erroneus@gmail.com',
    license='GPL',
    dependency_links=['git+https://github.com/Theano/Theano.git#egg=Theano'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='deep learning neuroimaging',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
)
