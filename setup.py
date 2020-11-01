from torchbayes import __version__
from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='torchbayes',
    version=__version__,
    author='EliaCereda',
    author_email='eliacereda@gmail.com',
    description='PyTorch library for Bayesian Deep Learning',
    keywords=['pytorch', 'bayesian-deep-learning', 'bayesian-neural-networks', 'bayes-by-backprop'],
    long_description_content_type="text/markdown",
    long_description=README,
    url='https://eliacereda.github.io/torchbayes',
    download_url='https://pypi.org/project/torchbayes/',
    python_requires='>=3.8',
    install_requires=['torch'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages(),
)

if __name__ == '__main__':
    setup(**setup_args)
