from setuptools import setup
import sys

try:
    import cython
except ModuleNotFoundError:
    raise RuntimeError('CytoPy requires that Cython>=0.27 be installed.')
if sys.version_info.major >= 3.6:
    raise RuntimeError('CytoPy requires Python version >= 3.6')
setup(
    name='CytoPy',
    version='0.0.1',
    packages=['CytoPy', 'CytoPy.flow', 'CytoPy.flow.gating', 'CytoPy.flow.clustering',
              'CytoPy.flow.supervised', 'CytoPy.tests', 'CytoPy.tests.data',
              'CytoPy.tests.test_clustering', 'CytoPy.tests.test_data', 'CytoPy.tests.test_flow'],
    url='https://github.com/burtonrj/CytoPy',
    license='MIT',
    author='Ross Burton',
    author_email='burtonrj@cardiff.ac.uk',
    description='Python framework for data-centric autonomous cytometry analysis',
    install_requires=open("requirements.txt").read(),
    dependency_links=['https://github.com/jacoblevine/PhenoGraph.git#egg=PhenoGraph',
                      'https://github.com/burtonrj/FlowUtilsPandas.git#egg=flowutilspd']
)
