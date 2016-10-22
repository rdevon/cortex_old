'''
Setup for Cortex
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
    version='0.3a',
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
    install_requires=[
        'nibabel', 'nipy', 'scikit-learn==0.17.0', 'scipy>0.17.1',
        'python-igraph', 'Pillow', 'terminaltables', 'matplotlib',
        'nilearn', 'colorclass', 'pyyaml', 'progressbar', 'mpld3', 'jinja2'],
    entry_points={
        'console_scripts': [
            'cortex-setup=cortex:main',
            'cortex-run=cortex.utils.trainer:main',
            'cortex-classifier-demo=cortex.demos.demos_basic:run_classifier_demo',
            'cortex-rbm-demo=cortex.demos.demos_basic:run_rbm_demo',
            'cortex-vae-demo=cortex.demos.demos_basic:run_vae_demo',
            'cortex-rbm-vbm-demo=cortex.demos.demos_neuroimaging:run_rbm_vbm_demo',
            'cortex-rbm-olin-demo=cortex.demos.demos_neuroimaging:run_rbm_olin_demo',
            'cortex-read-mri=cortex.analysis.load_mri:main',
            'cortex-read-fmri=cortex.analysis.load_fmri:main'
        ]
    },
    data_files=[
        'cortex/demos/demos_basic/yamls/classifier_mnist.yaml',
        'cortex/demos/demos_basic/yamls/rbm_mnist.yaml',
        'cortex/demos/demos_basic/yamls/vae_mnist.yaml',
        'cortex/demos/demos_neuroimaging/rbm_vbm.yaml',
        'cortex/demos/demos_neuroimaging/rbm_olin.yaml'
    ]
)
