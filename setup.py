from setuptools import setup, find_packages

setup(
    name='bnnbench',
    version='0.0.1',
    description='Benchmarking framework for BNNs as surrogate models for bayesian optimization.',
    author='Archit Bansal',
    author_email='archit.bansal.93@gmail.com',
    url="https://github.com/NeoChaos/bnnbench",
    license='BSD 3-Clause License',
    classifiers=['Development Status :: 4 - Beta'],
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['torch', 'torchvision', 'numpy', 'emcee', 'scipy', 'tensorboard', 'pandas', 'seaborn',
                      'matplotlib', 'ConfigSpace'],
    extras_require={},
    keywords=['python', 'Bayesian', 'neural networks', 'benchmarking', 'optimization'],
)
