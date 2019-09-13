from setuptools import setup, Extension
try:
    from numpy import get_include
except ImportError:
    raise RuntimeError(
        "NumPy is required to build the C extension in FlowUtils")


logicle_extension = Extension(
    'flowutils.logicle_c',
    sources=[
        'flowutils/logicle_c_ext/_logicle.c',
        'flowutils/logicle_c_ext/logicle.c'
    ],
    include_dirs=[get_include(), 'flowutils/logicle_c_ext'],
    extra_compile_args=['-std=c99']
)

setup(
    name='FlowUtils',
    version='0.7.1',
    packages=['flowutils'],
    package_data={'': []},
    description='Flow Cytometry Standard Utilities',
    license='BSD',
    ext_modules=[logicle_extension],
    install_requires=['numpy>=1.7']
)
