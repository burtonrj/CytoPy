from setuptools import setup

setup(
    name='CytoPy',
    version='0.0.1',
    packages=['flow', 'flow.gating', 'flow.gating.plotting', 'flow.clustering', 'flow.supervised', 'tests'],
    package_dir={'': 'cytopy'},
    url='https://github.com/burtonrj/CytoPy',
    license='MIT',
    author='Ross Burton',
    author_email='burtonrj@cardiff.ac.uk',
    description='Python framework for data-centric autonomous cytometry analysis'
)
