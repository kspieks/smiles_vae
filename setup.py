import os

from setuptools import find_packages, setup

__version__ = None

# utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="SMILES_VAE",
    version=__version__,
    author="Kevin Spiekermann",
    description="This codebase uses VAEs to generate SMILES strings.",
    url="https://github.com/kspieks/smiles_vae",
    packages=find_packages(),
    long_description=read('README.md'),
)
