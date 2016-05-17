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

    
setup(
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
    entry_points = {
        'console_scripts': ['cortex-setup=cortex:main']
    }
)
