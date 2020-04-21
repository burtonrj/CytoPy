from setuptools import setup, dist
import sys

if sys.version_info.major != 3:
    raise RuntimeError('CytoPy requires Python 3')
dist.Distribution().fetch_build_eggs(['Cython >=0.27'])
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
