from setuptools import setup
import sys

try:
    import cython
except ModuleNotFoundError:
    raise RuntimeError('CytoPy requires that Cython>=0.27 be installed.')
if sys.version_info.major >= 3.7:
    raise RuntimeError('CytoPy requires Python version >= 3.7')
setup(
    name='CytoPy',
    version='1.0.0',
    packages=['CytoPy',
              'CytoPy.flow',
              'CytoPy.flow.clustering', 
              'CytoPy.data',
              'CytoPy.tests',
              'CytoPy.tests/assets'],
    url='https://github.com/burtonrj/CytoPy',
    license='MIT',
    author='Ross Burton',
    author_email='burtonrj@cardiff.ac.uk',
    description='Python framework for data-centric autonomous cytometry analysis',
    install_requires=open("requirements.txt").read(),
    dependency_links=['https://github.com/jacoblevine/PhenoGraph.git#egg=PhenoGraph']
)
