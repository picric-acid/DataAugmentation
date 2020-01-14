from setuptools import setup, find_packages

readme = './README.md'
license = './LICENSE'

setup(
    name='DataAugmentation',
    version='0.0.1',
    description='Data Augmentation Package for ML',
    long_description=readme,
    author='picric_acid',
    author_email='picric_acid',
    install_requires=['numpy', 'scikit-learn'],
    url='https://github.com/picric-acid/DataAugmentation.git',
    license=license,
    packages=find_packages()
)