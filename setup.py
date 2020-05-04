from setuptools import setup, find_packages

setup(
    name='pybnn',
    version='0.1.0',
    description='Extension to pybnn by Aaron Klein and Moritz Freidank.',
    author='Archit Bansal, Aaron Klein, Moritz Freidank',
    author_email='bansala@cs.uni-freiburg.de',
    url="https://github.com/NeoChaos/pybnn",
    license='BSD 3-Clause License',
    classifiers=['Development Status :: 4 - Beta'],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['torch', 'torchvision', 'numpy', 'emcee', 'scipy', 'tensorboard'],
    extras_require={},
    keywords=['python', 'Bayesian', 'neural networks'],
)
