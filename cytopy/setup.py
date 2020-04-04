from setuptools import setup
import sys

if sys.version_info.major != 3:
    raise RuntimeError('CytoPy requires Python 3')

setup(
    name='CytoPy',
    version='0.0.1',
    packages=['flow', 'flow.gating', 'flow.gating.plotting', 'flow.clustering', 'flow.supervised', 'tests'],
    package_dir={'': 'cytopy'},
    url='https://github.com/burtonrj/CytoPy',
    license='MIT',
    author='Ross Burton',
    author_email='burtonrj@cardiff.ac.uk',
    description='Python framework for data-centric autonomous cytometry analysis',
    install_requires=open("requirements.txt").read()
)
