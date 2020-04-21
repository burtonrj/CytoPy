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
    packages=['flow', 'flow.gating', 'flow.clustering', 'flow.supervised', 'tests', 'tests.data',
              'tests.test_clustering', 'tests.test_data', 'tests.test_flow'],
    url='https://github.com/burtonrj/CytoPy',
    license='MIT',
    author='Ross Burton',
    author_email='burtonrj@cardiff.ac.uk',
    description='Python framework for data-centric autonomous cytometry analysis',
    install_requires=open("requirements.txt").read(),
    dependency_links=['https://github.com/jacoblevine/PhenoGraph.git#egg=PhenoGraph',
                      'https://github.com/burtonrj/FlowUtilsPandas.git#egg=flowutilspd']
)
