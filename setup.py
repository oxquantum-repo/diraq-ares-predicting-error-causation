from setuptools import setup

setup(
    name='barnaby',
    version='0.1.0',
    description='A package for analysing and quantifying the SPAM + state flip errors in a quantum computer',
    url='https://github.com/oxquantum-repo/diraq-ares-predicting-error-causation',
    author='Barnaby van Straaten, Brandon Severin',
    author_email='barnaby.vanstraaten@kellogg.ox.ac.uk',
    license='MIT',
    packages=['src'],
    install_requires=[
        'matplotlib',
        'numpy',
        'tqdm',
        'hmmlearn',
        'numdifftools'
    ],
)