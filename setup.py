from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torchnlp',
    version='0.1.0',
    description='Easy to use NLP library built on PyTorch and TorchText',
    long_description=long_description,
    url='https://github.com/kolloldas/torchnlp',
    author='Kollol Das',
    packages=find_packages(exclude=['tests'])
)

