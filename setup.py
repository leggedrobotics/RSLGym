from setuptools import find_packages
from distutils.core import setup


setup(
    name='rslgym',
    version='0.3.0',
    author='Takahiro Miki',
    license="MIT",
    packages=find_packages(),
    package_data={'rslgym': ['wrapper/src/*.cpp', 'wrapper/include/*.hpp', 'wrapper/setup.py', 'wrapper/CMakeLists.txt']},
    author_email='takahiro.miki1992@gmail.com',
    description='RSL version of raisim gym.',
    long_description='',
    install_requires=['gym==0.22.0', 'ruamel.yaml', 'numpy', 'termcolor', 'cloudpickle', 'tensorboard'],
    scripts=['bin/rslgym'],
)
